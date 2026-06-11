import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ConditionalGateRouter(nn.Module):
    def __init__(self, in_ch: int, num_experts: int = 3, reduction: int = 4):
        super().__init__()
        mid = max(in_ch // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_ch, mid),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(mid, num_experts),
        )
        self.sparsemax = Sparsemax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparsemax(self.gate(x))


class SharedStem(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class FourierGateAttention(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_heads: int = 4,
        qk_ratio: float = 0.5,
        freq_mode: str = "low",
        freq_sigma: float = 0.4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.freq_mode = freq_mode
        self.freq_sigma = freq_sigma

        self.d_qk = max(int(in_ch * qk_ratio), num_heads)
        self.d_qk = (self.d_qk // num_heads) * num_heads
        self.head_dim = self.d_qk // num_heads

        self.q_proj = nn.Conv2d(in_ch, self.d_qk, 1, bias=False)
        self.k_proj = nn.Conv2d(in_ch, self.d_qk, 1, bias=False)
        self.v_proj = nn.Conv2d(in_ch, in_ch, 1, bias=False)

        self.freq_scale = nn.Parameter(torch.ones(1, num_heads, 1, 1, 1) * 0.1)
        self.attn_norm = nn.BatchNorm2d(self.d_qk)

        self.gate_proj = nn.Sequential(
            nn.Conv2d(self.d_qk, in_ch, 1, bias=True),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.register_buffer('_freq_mask', None, persistent=False)
        self._cached_HW = (0, 0)

    def _get_freq_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        if self._cached_HW != (H, W) or self._freq_mask is None:
            freq_h = torch.fft.fftfreq(H, device=device)
            freq_w = torch.fft.rfftfreq(W, device=device)
            dist = torch.sqrt(freq_h.unsqueeze(1) ** 2 + freq_w.unsqueeze(0) ** 2)
            dist = dist / (dist.max() + 1e-8)
            mask_lf = torch.exp(-dist ** 2 / (2 * self.freq_sigma ** 2))
            mask = mask_lf if self.freq_mode == "low" else (1.0 - mask_lf)
            self._freq_mask = mask.reshape(1, 1, 1, H, W // 2 + 1)
            self._cached_HW = (H, W)
        return self._freq_mask

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape

        Q = self.q_proj(feat).reshape(B, self.num_heads, self.head_dim, H, W)
        K = self.k_proj(feat).reshape(B, self.num_heads, self.head_dim, H, W)
        V = self.v_proj(feat)

        _dtype = Q.dtype
        Q_f32 = Q.float()
        K_f32 = K.float()

        Q_freq = torch.fft.rfft2(Q_f32, norm="ortho")
        K_freq = torch.fft.rfft2(K_f32, norm="ortho")
        attn_freq = Q_freq * torch.conj(K_freq)
        attn_freq = attn_freq * self._get_freq_mask(H, W, feat.device)
        attn_freq = attn_freq * self.freq_scale.float()

        attn_spatial = torch.fft.irfft2(attn_freq, s=(H, W), norm="ortho")
        attn_spatial = attn_spatial.to(_dtype)
        attn_spatial = attn_spatial.reshape(B, self.d_qk, H, W)
        attn_spatial = self.attn_norm(attn_spatial)

        gate = self.gate_proj(attn_spatial)
        out = gate * V + V

        return self.out_proj(out)


class AgentAttention(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_heads: int = 4,
        qk_ratio: float = 0.5,
        num_agents: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_agents = num_agents

        self.d_qk = max(int(in_ch * qk_ratio), num_heads)
        self.d_qk = (self.d_qk // num_heads) * num_heads
        self.head_dim = self.d_qk // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(in_ch, self.d_qk, 1, bias=False)
        self.k_proj = nn.Conv2d(in_ch, self.d_qk, 1, bias=False)
        self.v_proj = nn.Conv2d(in_ch, self.d_qk, 1, bias=False)

        self.agent_tokens = nn.Parameter(
            torch.randn(1, num_heads, num_agents, self.head_dim) * 0.02
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(self.d_qk, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        h = self.num_heads
        d = self.head_dim
        N = H * W

        Q = self.q_proj(feat).reshape(B, h, d, N).permute(0, 1, 3, 2)
        K = self.k_proj(feat).reshape(B, h, d, N).permute(0, 1, 3, 2)
        V = self.v_proj(feat).reshape(B, h, d, N).permute(0, 1, 3, 2)

        A = self.agent_tokens.expand(B, -1, -1, -1)

        agent_attn = torch.softmax(
            (A @ K.transpose(-2, -1)) * self.scale, dim=-1
        )
        agent_ctx = agent_attn @ V

        query_attn = torch.softmax(
            (Q @ A.transpose(-2, -1)) * self.scale, dim=-1
        )
        out = query_attn @ agent_ctx

        out = out.permute(0, 1, 3, 2).reshape(B, self.d_qk, H, W)

        return self.out_proj(out)


class GlobalExpertGroup(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_heads: int = 4,
        num_agents: int = 16,
        freq_sigma: float = 0.4,
        qk_ratio: float = 0.5,
    ):
        super().__init__()
        self.out_ch = out_ch

        self.stem = SharedStem(in_ch)

        self.experts = nn.ModuleList([
            FourierGateAttention(
                in_ch, out_ch,
                num_heads=num_heads,
                qk_ratio=qk_ratio,
                freq_mode="low",
                freq_sigma=freq_sigma,
            ),
            FourierGateAttention(
                in_ch, out_ch,
                num_heads=num_heads,
                qk_ratio=qk_ratio,
                freq_mode="high",
                freq_sigma=freq_sigma,
            ),
            AgentAttention(
                in_ch, out_ch,
                num_heads=num_heads,
                qk_ratio=qk_ratio,
                num_agents=num_agents,
            ),
        ])

        self.router = ConditionalGateRouter(in_ch, num_experts=3)

        self.use_residual = (in_ch == out_ch)
        if in_ch != out_ch:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.router(x)
        feat = self.stem(x)

        out = 0
        for i, expert in enumerate(self.experts):
            w_i = weights[:, i].reshape(-1, 1, 1, 1)
            e_out = expert(feat)
            out = out + w_i * e_out

        if self.use_residual:
            out = out + x
        elif hasattr(self, "residual_proj"):
            out = out + self.residual_proj(x)

        return out