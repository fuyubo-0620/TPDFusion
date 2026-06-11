import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIPTextInjector(nn.Module):
    CLIP_DIM = 512

    def __init__(
        self,
        proj_dim: int = 512,
        clip_model: str = "ViT-B/32",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.clip_model, _ = clip.load(clip_model, device="cpu")
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        self.proj_shortcut = nn.Linear(self.CLIP_DIM, proj_dim, bias=False)

        self.proj_deep = nn.Sequential(
            nn.Linear(self.CLIP_DIM, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

        self.proj_ln = nn.LayerNorm(proj_dim)

    def _encode(self, texts: list, device: torch.device) -> torch.Tensor:
        with torch.no_grad():
            tokens = clip.tokenize(texts, truncate=True).to(device)
            feats = self.clip_model.encode_text(tokens).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def forward(
        self,
        texts: list,
        device: torch.device,
    ) -> torch.Tensor:
        raw_feat = self._encode(texts, device)
        text_feat = self.proj_ln(
            self.proj_shortcut(raw_feat) + self.proj_deep(raw_feat)
        )
        return text_feat