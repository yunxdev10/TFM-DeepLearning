from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (32, 64, 128, 256),
    ):
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = in_channels
        for feature in features:
            self.down_blocks.append(DoubleConv(channels, feature))
            channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for feature in reversed(features):
            self.up_transpose.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.up_blocks.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.up_blocks)):
            x = self.up_transpose[idx](x)
            skip = skip_connections[idx]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.up_blocks[idx](x)

        return self.final_conv(x)


class AttentionGate(nn.Module):
    def __init__(self, gating_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.gating_proj = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gating: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gating.shape[-2:] != skip.shape[-2:]:
            gating = F.interpolate(gating, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.attention(self.relu(self.gating_proj(gating) + self.skip_proj(skip)))
        return skip * attention


class AttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (32, 64, 128, 256),
    ):
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = in_channels
        for feature in features:
            self.down_blocks.append(DoubleConv(channels, feature))
            channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.up_transpose = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for feature in reversed(features):
            self.up_transpose.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.attention_gates.append(AttentionGate(feature, feature, max(feature // 2, 1)))
            self.up_blocks.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.up_blocks)):
            x = self.up_transpose[idx](x)
            skip = skip_connections[idx]
            gated_skip = self.attention_gates[idx](x, skip)
            if x.shape[-2:] != gated_skip.shape[-2:]:
                x = F.interpolate(x, size=gated_skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat((gated_skip, x), dim=1)
            x = self.up_blocks[idx](x)

        return self.final_conv(x)


def build_segmentation_model(
    architecture: str,
    in_channels: int,
    out_channels: int = 1,
    base_features: int = 32,
) -> nn.Module:
    features = (base_features, base_features * 2, base_features * 4, base_features * 8)
    architecture = architecture.lower().replace("-", "_")
    if architecture in {"unet", "u_net"}:
        return UNet(in_channels=in_channels, out_channels=out_channels, features=features)
    if architecture in {"attention_unet", "attention_u_net", "att_unet"}:
        return AttentionUNet(in_channels=in_channels, out_channels=out_channels, features=features)
    raise ValueError(f"Unsupported segmentation architecture: {architecture}")
