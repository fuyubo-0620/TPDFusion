import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from wtconv.wtconv2d import WTConv2d


class Sparsemax(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        z_sorted, _ = z.sort(dim=dim, descending=True)
        z_cumsum = z_sorted.cumsum(dim=dim)
        N = z.size(dim)
        rhos = torch.arange(1, N + 1, device=z.device, dtype=z.dtype)
        shape = [1] * z.dim()
        shape[dim] = N
        rhos = rhos.view(shape)
        support = (1 + rhos * z_sorted > z_cumsum)
        k = support.sum(dim=dim, keepdim=True)
        k_idx = (k - 1).clamp(min=0).long()
        tau_cumsum = z_cumsum.gather(dim, k_idx)
        tau = (tau_cumsum - 1.0) / k.to(z.dtype)
        return (z - tau).clamp(min=0)


class FreqSpatialSparseRouter(nn.Module):
    def __init__(self, in_ch: int, num_experts: int = 3, pool_size: int = 5):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch // 2, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_ch // 2, num_experts, kernel_size=1, bias=True),
        )
        self.sparsemax = Sparsemax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_cue = self.spatial(x)
        low_freq = self.avg_pool(x)
        high_freq = torch.abs(x - low_freq)
        combined = torch.cat([spatial_cue, high_freq], dim=1)
        logits = self.gate(combined)
        weights = self.sparsemax(logits)
        return weights


class WaveletConvExpert(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 5,
        wt_levels: int = 2,
        wt_type: str = "haar",
    ):
        super().__init__()
        self.wtconv = WTConv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=1,
            bias=False,
            wt_levels=wt_levels,
            wt_type=wt_type,
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.wtconv2 = WTConv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=1,
            bias=False,
            wt_levels=wt_levels,
            wt_type=wt_type,
        )
        self.bn1b = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.wtconv(x)))
        x = self.act(self.bn1b(self.wtconv2(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class DCNExpert(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 4):
        super().__init__()
        self.n_points = 9
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_ch, self.n_points * 3, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)
        self.deform_conv = DeformConv2d(
            in_ch, in_ch,
            kernel_size=3, padding=1,
            groups=groups, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_mask = self.offset_conv(x)
        offset = offset_mask[:, :self.n_points * 2, :, :]
        mask = torch.sigmoid(offset_mask[:, self.n_points * 2:, :, :])
        x = self.act(self.bn1(self.deform_conv(x, offset, mask)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class LocalExpertGroup(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dcn_groups: int = 4,
        wt_levels: int = 1,
        wt_type: str = "db1",
    ):
        super().__init__()
        self.experts = nn.ModuleList([
            WaveletConvExpert(in_ch, out_ch, kernel_size=5, wt_levels=wt_levels, wt_type=wt_type),
            WaveletConvExpert(in_ch, out_ch, kernel_size=7, wt_levels=wt_levels, wt_type=wt_type),
            DCNExpert(in_ch, out_ch, groups=dcn_groups),
        ])
        self.router = FreqSpatialSparseRouter(in_ch, num_experts=3)
        self.use_residual = (in_ch == out_ch)
        if in_ch != out_ch:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.router(x)
        out = None
        for i, expert in enumerate(self.experts):
            w_i = weights[:, i:i + 1, :, :]
            if w_i.max().item() == 0:
                continue
            e_out = expert(x)
            out = w_i * e_out if out is None else out + w_i * e_out
        if out is None:
            out = self.experts[0](x)
        if self.use_residual:
            out = out + x
        elif hasattr(self, "residual_proj"):
            out = out + self.residual_proj(x)
        return out