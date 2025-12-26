import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet3DMin(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = conv_block(1, base)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = conv_block(base, base*2)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = conv_block(base*2, base*4)

        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.dec2 = conv_block(base*4, base*2)

        self.up1 = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.dec1 = conv_block(base*2, base)

        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)          # (B, base, D, H, W)
        p1 = self.pool1(e1)        # /2

        e2 = self.enc2(p1)         # (B, base*2, ...)
        p2 = self.pool2(e2)        # /4

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        # handle odd shapes just in case
        if u2.shape[-3:] != e2.shape[-3:]:
            u2 = F.interpolate(u2, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[-3:] != e1.shape[-3:]:
            u1 = F.interpolate(u1, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        y = self.out(d1)
        # Since CT is 0..1, clamp to keep stable early training
        return torch.sigmoid(y)
