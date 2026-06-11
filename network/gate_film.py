import torch
import torch.nn as nn


class GatedFiLMModulation(nn.Module):
    def __init__(self, text_feat_dim: int = 512, img_feat_dim: int = 64):
        super().__init__()

        self.fc_gamma = nn.Linear(text_feat_dim, img_feat_dim)
        self.fc_beta = nn.Linear(text_feat_dim, img_feat_dim)

        self.bn = nn.BatchNorm2d(img_feat_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        self.spatial_kernel_gen = nn.Sequential(
            nn.Linear(text_feat_dim, img_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(img_feat_dim, img_feat_dim),
        )
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(8, 1, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )

        nn.init.normal_(self.fc_gamma.weight, std=0.02)
        nn.init.ones_(self.fc_gamma.bias)

        nn.init.normal_(self.fc_beta.weight, std=0.02)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(
        self,
        img_feat: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = img_feat.shape

        gamma = self.fc_gamma(text_feat).unsqueeze(-1).unsqueeze(-1)
        beta = self.fc_beta(text_feat).unsqueeze(-1).unsqueeze(-1)
        modulated = gamma * img_feat + beta

        kernel = self.spatial_kernel_gen(text_feat).unsqueeze(-1).unsqueeze(-1)
        spatial_score = (img_feat * kernel).sum(dim=1, keepdim=True)
        spatial_attn = self.spatial_refine(spatial_score)
        modulated = modulated * spatial_attn + modulated * (1.0 - spatial_attn) * 0.5

        modulated = self.bn(modulated)

        return img_feat + self.residual_scale * (modulated - img_feat)