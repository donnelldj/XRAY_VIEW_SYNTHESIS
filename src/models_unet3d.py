# src/models_unet3d.py
# 3D U-Net for volume-to-volume regression (BP volume -> CT volume)
# Designed for small batch sizes (batch_size=1) using GroupNorm.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_groups(num_channels: int, max_groups: int = 8) -> int:
    """
    Choose a GroupNorm group count that divides num_channels.
    Prefer larger group counts up to max_groups, but fall back cleanly.
    """
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return max(g, 1)


class ConvGNAct3D(nn.Module):
    """Conv3D -> GroupNorm -> SiLU"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        groups = _safe_groups(out_ch, max_groups=8)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.gn = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class DoubleConv3D(nn.Module):
    """Two ConvGNAct3D blocks."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct3D(in_ch, out_ch),
            ConvGNAct3D(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3D(nn.Module):
    """Downsample (stride-2 maxpool) + DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up3D(nn.Module):
    """
    Upsample + concat skip + DoubleConv.
    Uses trilinear upsample (stable, less artifacts than transpose conv).
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv3D(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample x to skip spatial dims (D, H, W)
        x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D_CT(nn.Module):
    """
    3D U-Net for BP->CT regression.

    Input:  [B, in_ch, D, H, W]
    Output: [B, out_ch, D, H, W]

    Tips:
      - For RTX 4050 / limited VRAM: use base=16, batch_size=1
      - GroupNorm makes training stable with batch_size=1
      - If you later want more capacity, base=24 or base=32 (watch OOM)
    """
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base: int = 16,
        depth: int = 4,
    ):
        """
        depth=4 gives channel ladder:
          base, 2b, 4b, 8b, 16b (bottleneck)
        """
        super().__init__()
        assert depth in (3, 4), "Use depth=3 or depth=4 for sanity/VRAM."

        b = base
        self.inc = DoubleConv3D(in_ch, b)

        if depth == 3:
            self.down1 = Down3D(b, 2 * b)
            self.down2 = Down3D(2 * b, 4 * b)
            self.down3 = Down3D(4 * b, 8 * b)

            self.up1 = Up3D(8 * b, 4 * b, 4 * b)
            self.up2 = Up3D(4 * b, 2 * b, 2 * b)
            self.up3 = Up3D(2 * b, b, b)

            self.outc = nn.Conv3d(b, out_ch, kernel_size=1)

        else:  # depth == 4
            self.down1 = Down3D(b, 2 * b)
            self.down2 = Down3D(2 * b, 4 * b)
            self.down3 = Down3D(4 * b, 8 * b)
            self.down4 = Down3D(8 * b, 16 * b)

            self.up1 = Up3D(16 * b, 8 * b, 8 * b)
            self.up2 = Up3D(8 * b, 4 * b, 4 * b)
            self.up3 = Up3D(4 * b, 2 * b, 2 * b)
            self.up4 = Up3D(2 * b, b, b)

            self.outc = nn.Conv3d(b, out_ch, kernel_size=1)

        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)       # b

        x2 = self.down1(x1)    # 2b
        x3 = self.down2(x2)    # 4b
        x4 = self.down3(x3)    # 8b

        if self.depth == 3:
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            return self.outc(x)

        x5 = self.down4(x4)    # 16b bottleneck

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


if __name__ == "__main__":
    # quick sanity check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet3D_CT(in_ch=1, out_ch=1, base=16, depth=4).to(device)
    x = torch.randn(1, 1, 128, 128, 128, device=device)  # adjust to match your dataset
    with torch.no_grad():
        y = model(x)
    print("in :", tuple(x.shape))
    print("out:", tuple(y.shape))
