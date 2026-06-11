import torch
import torch.nn as nn
import torch.nn.functional as F

from network.Local_expert import LocalExpertGroup as LocalExpert
from network.Global_Expert import GlobalExpertGroup as GlobalExpert
from network.Encoder_Decoder import Encoder, Decoder
from network.Crossmodal_inter import FocusedLinearCrossAttention
from network.Fusion_Gate import FusionGate
from network.CLIP_Injector import CLIPTextInjector
from network.SegHead import SegFormerHead


class WaveFusionNet(nn.Module):
    def __init__(
        self,
        base_channels: int = 48,
        use_text: bool = True,
        text_proj_dim: int = 512,
        clip_model: str = "ViT-B/32",
        use_task_head: bool = True,
        num_classes: int = 9,
        seg_embed_dim: int = 128,
        seg_drop_rate: float = 0.1,
        seg_align_index: int = 2,
    ):
        super().__init__()
        self.use_text = use_text
        self.use_task_head = use_task_head

        self.encoder = Encoder(in_channels=2, base_channels=base_channels)

        self.local_expert_1 = LocalExpert(in_ch=64, out_ch=64)
        self.global_expert_1 = GlobalExpert(in_ch=64, out_ch=64)
        self.cross_fuse_1 = FocusedLinearCrossAttention(dim=64, num_heads=4)

        self.local_expert_2 = LocalExpert(in_ch=64, out_ch=96)
        self.global_expert_2 = GlobalExpert(in_ch=64, out_ch=96)
        self.cross_fuse_2 = FocusedLinearCrossAttention(dim=96, num_heads=4)

        self.local_expert_3 = LocalExpert(in_ch=96, out_ch=64)
        self.global_expert_3 = GlobalExpert(in_ch=96, out_ch=64)

        self.fusion_gate = FusionGate(
            feat_ch=64,
            num_blocks=3,
            text_feat_dim=text_proj_dim,
            use_text=use_text,
        )

        self.decoder = Decoder(
            deep_ch=64,
            skip_channels=(64, 48, 48),
            base_channels=base_channels,
            text_feat_dim=text_proj_dim,
            use_text=use_text,
        )

        if use_text:
            self.text_injector = CLIPTextInjector(
                proj_dim=text_proj_dim,
                clip_model=clip_model,
            )

        if use_task_head:
            self.seg_head = SegFormerHead(
                in_channels_list=[base_channels, base_channels, 64, 64],
                embed_dim=seg_embed_dim,
                num_classes=num_classes,
                drop_rate=seg_drop_rate,
                align_index=seg_align_index,
            )

    def forward(
        self,
        vis: torch.Tensor,
        ir: torch.Tensor,
        texts: list = None,
        output_seg: bool = True,
    ):
        device = vis.device

        text_feat = None
        if self.use_text and texts is not None:
            text_feat = self.text_injector(texts, device)

        enc_skip_0, enc_skip_1, enc_skip_2, feat_down = self.encoder(vis, ir)

        local_1 = self.local_expert_1(feat_down)
        global_1 = self.global_expert_1(feat_down)
        local_1, global_1 = self.cross_fuse_1(local_1, global_1)

        local_2 = self.local_expert_2(local_1)
        global_2 = self.global_expert_2(global_1)
        local_2, global_2 = self.cross_fuse_2(local_2, global_2)

        local_3 = self.local_expert_3(local_2)
        global_3 = self.global_expert_3(global_2)

        deep_feat, feat_vec = self.fusion_gate(
            local_3, global_3, text_feat=text_feat,
        )

        seg_logits = None
        if self.use_task_head and output_seg:
            seg_features = [enc_skip_0, enc_skip_1, enc_skip_2, deep_feat]
            seg_logits = self.seg_head(
                features=seg_features,
                target_size=(vis.shape[2], vis.shape[3]),
            )

        fused = self.decoder(
            deep_feat=deep_feat,
            enc_skip_2=enc_skip_2,
            enc_skip_1=enc_skip_1,
            enc_skip_0=enc_skip_0,
            text_feat=text_feat,
        )

        if self.use_text and feat_vec is not None:
            return fused, feat_vec, text_feat, seg_logits
        else:
            return fused, None, None, seg_logits