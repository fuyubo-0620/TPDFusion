import torch
import torch.nn as nn
import torch.nn.functional as F


class FocusedLinearCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qk_ratio: float = 0.5,
        init_gamma: float = 0.01,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_v = dim // num_heads
        self.d_qk = max(int(dim * qk_ratio) // num_heads, 4) * num_heads
        self.d_qk_head = self.d_qk // num_heads

        self.q_proj = nn.Conv2d(dim, self.d_qk, 1, bias=False)
        self.k_proj = nn.Conv2d(dim, self.d_qk, 1, bias=False)

        self.v_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.focus_gate = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )

        mid = max(dim // 4, 16)
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.gamma_l = nn.Parameter(torch.full((1,), init_gamma))
        self.gamma_g = nn.Parameter(torch.full((1,), init_gamma))

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, alpha=1.0) + 1.0

    def _focused_linear_attn(
        self,
        q_feat: torch.Tensor,
        kv_feat: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = q_feat.shape
        N = H * W
        h = self.num_heads
        d_q = self.d_qk_head
        d_v = self.d_v

        Q = self.q_proj(q_feat)
        K = self.k_proj(kv_feat)
        V = self.v_proj(kv_feat)

        gate = self.focus_gate(kv_feat)

        if self.d_qk == C:
            K = K * gate
        else:
            gate_k = F.adaptive_avg_pool1d(
                gate.flatten(2).mean(dim=2, keepdim=True).permute(0, 2, 1),
                self.d_qk,
            ).permute(0, 2, 1).unsqueeze(-1)
            gate_spatial = gate.mean(dim=1, keepdim=True)
            K = K * gate_spatial
        V = V * gate

        Q = Q.reshape(B, h, d_q, N).permute(0, 1, 3, 2)
        K = K.reshape(B, h, d_q, N).permute(0, 1, 3, 2)
        V = V.reshape(B, h, d_v, N).permute(0, 1, 3, 2)

        Q = self._phi(Q)
        K = self._phi(K)

        KV = K.transpose(-2, -1) @ V
        out = Q @ KV

        K_sum = K.sum(dim=-2)
        Z = (Q * K_sum.unsqueeze(-2)).sum(dim=-1, keepdim=True)
        out = out / (Z + 1e-6)

        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        return self.out_proj(out)

    def forward(
        self,
        local_feat: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> tuple:
        from_global = self._focused_linear_attn(
            q_feat=local_feat, kv_feat=global_feat,
        )
        from_local = self._focused_linear_attn(
            q_feat=global_feat, kv_feat=local_feat,
        )

        local_out = local_feat + self.gamma_l * from_global
        global_out = global_feat + self.gamma_g * from_local

        return local_out, global_out