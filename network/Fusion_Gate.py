import torch
import torch.nn as nn

from network.gate_film import GatedFiLMModulation


class FusionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pw = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pw(x)
        out = self.dw(out)
        return out + x


class FusionGate(nn.Module):
    def __init__(
        self,
        feat_ch: int = 64,
        num_blocks: int = 3,
        text_feat_dim: int = 512,
        use_text: bool = True,
    ):
        super().__init__()
        self.use_text = use_text

        self.compress = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, 1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.blocks = nn.ModuleList([
            FusionBlock(feat_ch) for _ in range(num_blocks)
        ])

        if use_text:
            self.film_layers = nn.ModuleList([
                GatedFiLMModulation(
                    text_feat_dim=text_feat_dim,
                    img_feat_dim=feat_ch,
                )
                for _ in range(num_blocks)
            ])

            self.align_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feat_ch, text_feat_dim),
                nn.LayerNorm(text_feat_dim),
            )

    def forward(
        self,
        local_feat: torch.Tensor,
        global_feat: torch.Tensor,
        text_feat: torch.Tensor = None,
    ):
        x = torch.cat([local_feat, global_feat], dim=1)
        x = self.compress(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.use_text and text_feat is not None:
                x = self.film_layers[i](x, text_feat)

        feat_vec = None
        if self.use_text and text_feat is not None:
            feat_vec = self.align_proj(x)
        return x, feat_vec