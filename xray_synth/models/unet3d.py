from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """
    Conv3D -> GroupNorm -> SiLU (x2)

    Why GroupNorm:
    - stable for small batch sizes (common in 3D medical workloads)
    - avoids BatchNorm issues on tiny batches
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
            nn.SiLU(),
            nn.Conv3d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    """
    Small 3D UNet operating in latent space.

    Input:  (B,1,Zl,Yl,Xl)  backprojected+pooled volume
    Output: (B,1,Zl,Yl,Xl)  predicted CT latent (logits; caller typically sigmoid)
    """
    def __init__(self, base: int = 16):
        super().__init__()
        self.enc1 = ConvBlock3D(1, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(base * 2, base * 4)

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base)

        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)
