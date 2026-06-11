import torch
import torch.nn as nn
import torch.nn.functional as F

from network.gate_film import GatedFiLMModulation


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DWBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.use_residual = (in_ch == out_ch and stride == 1)
        if not self.use_residual and stride == 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        elif not self.use_residual and stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.dw(x)))
        out = self.bn2(self.pw(out))
        out = self.act(out + self.shortcut(x))
        return out


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(mid, channels),
        )

    def forward(self, x):
        avg = x.mean(dim=[2, 3])
        mx = x.amax(dim=[2, 3])
        gate = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = DWBlock(in_ch, out_ch, stride=2)
        self.refine = DWBlock(out_ch, out_ch, stride=1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.refine(self.down(x)) + self.shortcut(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 48):
        super().__init__()
        mid = base_channels // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.stage0_block1 = DWBlock(mid, base_channels, stride=1)
        self.stage0_block2 = DWBlock(base_channels, base_channels, stride=1)
        self.stage0_skip = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1, bias=False),
            nn.BatchNorm2d(base_channels),
        )
        self.stage1 = EncoderStage(base_channels, base_channels)
        self.stage2 = EncoderStage(base_channels, 64)
        self.stage3 = EncoderStage(64, 64)
        self.channel_gate = ChannelGate(64, reduction=4)
        self.apply(init_weights)

    def forward(self, vis, ir):
        x = torch.cat([vis, ir], dim=1)
        s0 = self.stem(x)
        s0 = self.stage0_block1(s0)
        s0 = self.stage0_block2(s0)
        enc_skip_0 = s0 + self.stage0_skip(x)
        enc_skip_1 = self.stage1(enc_skip_0)
        enc_skip_2 = self.stage2(enc_skip_1)
        feat_down = self.stage3(enc_skip_2)
        feat_down = self.channel_gate(feat_down)
        return enc_skip_0, enc_skip_1, enc_skip_2, feat_down


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.compress = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.refine = DWBlock(out_ch, out_ch, stride=1)

    def forward(self, x, skip):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.compress(x)
        x = self.refine(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        deep_ch: int = 64,
        skip_channels: tuple = (64, 48, 48),
        base_channels: int = 48,
        text_feat_dim: int = 512,
        use_text: bool = True,
    ):
        super().__init__()
        self.use_text = use_text
        s2_ch, s1_ch, s0_ch = skip_channels

        self.dec_block3 = DecoderBlock(in_ch=deep_ch, skip_ch=s2_ch, out_ch=64)
        self.dec_block2 = DecoderBlock(in_ch=64, skip_ch=s1_ch, out_ch=base_channels)
        self.dec_block1 = DecoderBlock(in_ch=base_channels, skip_ch=s0_ch, out_ch=base_channels)

        self.output_proj = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1),
            nn.Tanh(),
        )

        if use_text:
            dec_feat_dims = [64, base_channels, base_channels]
            self.dec_film_layers = nn.ModuleList([
                GatedFiLMModulation(
                    text_feat_dim=text_feat_dim,
                    img_feat_dim=dim,
                )
                for dim in dec_feat_dims
            ])

        self.apply(init_weights)

        last_conv = self.output_proj[-2]
        nn.init.kaiming_normal_(last_conv.weight, a=0.1, mode='fan_in',
                                nonlinearity='leaky_relu')
        nn.init.zeros_(last_conv.bias)

        if use_text:
            for film in self.dec_film_layers:
                nn.init.normal_(film.fc_gamma.weight, std=0.02)
                nn.init.ones_(film.fc_gamma.bias)

                nn.init.normal_(film.fc_beta.weight, std=0.02)
                nn.init.zeros_(film.fc_beta.bias)

                nn.init.constant_(film.residual_scale, 0.1)

    def forward(
        self,
        deep_feat: torch.Tensor,
        enc_skip_2: torch.Tensor,
        enc_skip_1: torch.Tensor,
        enc_skip_0: torch.Tensor,
        text_feat: torch.Tensor = None,
    ) -> torch.Tensor:

        x = self.dec_block3(deep_feat, enc_skip_2)
        if self.use_text and text_feat is not None:
            x = self.dec_film_layers[0](x, text_feat)

        x = self.dec_block2(x, enc_skip_1)
        if self.use_text and text_feat is not None:
            x = self.dec_film_layers[1](x, text_feat)

        x = self.dec_block1(x, enc_skip_0)
        if self.use_text and text_feat is not None:
            x = self.dec_film_layers[2](x, text_feat)

        fused = self.output_proj(x)
        return fused