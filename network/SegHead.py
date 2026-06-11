import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerHead(nn.Module):
    def __init__(
        self,
        in_channels_list: list = None,
        embed_dim: int = 128,
        num_classes: int = 9,
        drop_rate: float = 0.1,
        align_index: int = 2,
    ):
        super().__init__()

        if in_channels_list is None:
            in_channels_list = [48, 48, 64, 64]

        self.num_inputs = len(in_channels_list)
        self.align_index = align_index
        self.embed_dim = embed_dim

        self.linear_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for ch in in_channels_list
        ])

        self.fuse_proj = nn.Sequential(
            nn.Conv2d(embed_dim * self.num_inputs, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(drop_rate)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        features: list,
        target_size: tuple = None,
    ) -> torch.Tensor:
        assert len(features) == self.num_inputs

        align_feat = features[self.align_index]
        align_size = align_feat.shape[2:]

        projected = []
        for i, feat in enumerate(features):
            h = self.linear_projections[i](feat)
            if h.shape[2:] != align_size:
                h = F.interpolate(h, size=align_size, mode='bilinear', align_corners=False)
            projected.append(h)

        x = torch.cat(projected, dim=1)
        x = self.fuse_proj(x)

        x = self.dropout(x)
        seg_logits = self.classifier(x)

        if target_size is not None and seg_logits.shape[2:] != target_size:
            seg_logits = F.interpolate(seg_logits, size=target_size, mode='bilinear', align_corners=False)

        return seg_logits


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 9,
        class_weights: torch.Tensor = None,
        ignore_index: int = 255,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
        )

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid_mask = (target != self.ignore_index)
        target_clean = target.clone()
        target_clean[~valid_mask] = 0

        target_one_hot = F.one_hot(target_clean.long(), self.num_classes).permute(0, 3, 1, 2).float()
        valid_mask_4d = valid_mask.unsqueeze(1).float()

        pred = pred * valid_mask_4d
        target_one_hot = target_one_hot * valid_mask_4d

        smooth = 1e-5
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + smooth) / (union + smooth)

        return 1.0 - dice_score.mean()

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_labels: torch.Tensor,
    ) -> torch.Tensor:
        if seg_logits.shape[2:] != seg_labels.shape[1:]:
            seg_logits = F.interpolate(seg_logits, size=seg_labels.shape[1:], mode='bilinear', align_corners=False)

        loss_ce = self.ce_loss(seg_logits, seg_labels.long())

        if self.dice_weight > 0:
            pred_prob = F.softmax(seg_logits, dim=1)
            loss_dice = self._dice_loss(pred_prob, seg_labels)
            return loss_ce + self.dice_weight * loss_dice
        else:
            return loss_ce