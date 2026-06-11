import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBNSiLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DFL(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float32))

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.reshape(B, 4, self.reg_max, H, W)
        x = F.softmax(x, dim=2)
        x = (x * self.proj.reshape(1, 1, self.reg_max, 1, 1)).sum(dim=2)
        return x


class DecoupledHead(nn.Module):
    def __init__(
        self,
        in_ch: int,
        mid_ch: int = 64,
        num_classes: int = 3,
        num_convs: int = 2,
        reg_max: int = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.stem = ConvBNSiLU(in_ch, mid_ch, k=1, s=1, p=0)

        self.cls_convs = nn.Sequential(
            *[ConvBNSiLU(mid_ch, mid_ch) for _ in range(num_convs)]
        )
        self.cls_pred = nn.Conv2d(mid_ch, num_classes, 1)

        self.reg_convs = nn.Sequential(
            *[ConvBNSiLU(mid_ch, mid_ch) for _ in range(num_convs)]
        )
        self.reg_pred = nn.Conv2d(mid_ch, 4 * reg_max, 1)

        self.dfl = DFL(reg_max=reg_max)

        self._init_weights()

    def _init_weights(self):
        prior = 0.01
        nn.init.constant_(self.cls_pred.bias, -math.log((1 - prior) / prior))
        nn.init.constant_(self.reg_pred.bias, 0.0)

    def forward(self, feat):
        x = self.stem(feat)
        cls_logits = self.cls_pred(self.cls_convs(x))
        reg_raw = self.reg_pred(self.reg_convs(x))
        reg_decoded = self.dfl(reg_raw)
        return cls_logits, reg_raw, reg_decoded


class YOLOv8DetHead(nn.Module):
    def __init__(
        self,
        in_channels_list: list = None,
        mid_ch: int = 64,
        num_classes: int = 3,
        num_convs: int = 2,
        reg_max: int = 16,
        strides: tuple = (2, 4, 8),
    ):
        super().__init__()

        if in_channels_list is None:
            in_channels_list = [48, 64, 64]

        self.num_scales = len(in_channels_list)
        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max

        self.heads = nn.ModuleList([
            DecoupledHead(
                in_ch=in_channels_list[i],
                mid_ch=mid_ch,
                num_classes=num_classes,
                num_convs=num_convs,
                reg_max=reg_max,
            )
            for i in range(self.num_scales)
        ])

    def forward(self, features: list):
        assert len(features) == self.num_scales

        cls_scores = []
        reg_distri = []
        reg_bboxes = []

        for feat in features:
            cls_out, reg_raw, reg_dec = self.heads[features.index(feat)](feat)
            cls_scores.append(cls_out)
            reg_distri.append(reg_raw)
            reg_bboxes.append(reg_dec)

        if self.training:
            return cls_scores, reg_distri, reg_bboxes
        else:
            return self._decode(cls_scores, reg_bboxes)

    @torch.no_grad()
    def _decode(self, cls_scores, reg_bboxes):
        all_preds = []
        for i in range(self.num_scales):
            cls_out = cls_scores[i].sigmoid()
            ltrb = reg_bboxes[i]
            B, C, H, W = cls_out.shape
            stride = self.strides[i]

            yv, xv = torch.meshgrid(
                torch.arange(H, device=cls_out.device, dtype=cls_out.dtype),
                torch.arange(W, device=cls_out.device, dtype=cls_out.dtype),
                indexing="ij",
            )
            cx = (xv + 0.5) * stride
            cy = (yv + 0.5) * stride

            l, t, r, b = ltrb.unbind(dim=1)
            x1 = cx.unsqueeze(0) - l * stride
            y1 = cy.unsqueeze(0) - t * stride
            x2 = cx.unsqueeze(0) + r * stride
            y2 = cy.unsqueeze(0) + b * stride

            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            pred = torch.cat([boxes, cls_out], dim=1)
            pred = pred.flatten(2).permute(0, 2, 1)
            all_preds.append(pred)

        return torch.cat(all_preds, dim=1)

    def make_anchors(self, features, device):
        anchor_points, stride_tensor = [], []
        for i, feat in enumerate(features):
            _, _, H, W = feat.shape
            stride = self.strides[i]
            sy, sx = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing="ij",
            )
            points = torch.stack([sx + 0.5, sy + 0.5], dim=-1).reshape(-1, 2)
            anchor_points.append(points)
            stride_tensor.append(
                torch.full((H * W, 1), stride, device=device, dtype=torch.float32)
            )
        return torch.cat(anchor_points), torch.cat(stride_tensor)


def non_max_suppression(
    decoded,
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=300,
):
    from torchvision.ops import nms

    B = decoded.shape[0]
    results = []

    for b in range(B):
        pred = decoded[b]
        boxes = pred[:, :4]
        scores = pred[:, 4:]

        max_score, cls_id = scores.max(dim=-1)

        keep = max_score > conf_thres
        boxes = boxes[keep]
        max_score = max_score[keep]
        cls_id = cls_id[keep]

        if boxes.numel() == 0:
            results.append(torch.zeros(0, 6, device=decoded.device))
            continue

        offset = cls_id.float() * 4096
        boxes_offset = boxes + offset[:, None]

        nms_idx = nms(boxes_offset, max_score, iou_thres)
        nms_idx = nms_idx[:max_det]

        det = torch.cat([
            boxes[nms_idx],
            max_score[nms_idx, None],
            cls_id[nms_idx, None].float(),
        ], dim=-1)

        results.append(det)

    return results