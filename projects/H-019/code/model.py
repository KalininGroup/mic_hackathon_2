#!/usr/bin/env python
# encoding: UTF8

# Auto-encoder model.
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p)
        self.norm = nn.BatchNorm2d(out_ch) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, out_p=0, norm=True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, k, stride=s, padding=p, output_padding=out_p)
        self.norm = nn.BatchNorm2d(out_ch) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.deconv(x)))

class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape  # e.g., (256, 8, 8)
    def forward(self, x):
        return x.view(x.size(0), *self.shape)
        
class ConvAutoencoder(nn.Module):
    """
    Inputs:  x in [0,1] range, shape [B, C, H, W] with H and W divisible by 16.
    Outputs: reconstruction in [0,1], shape [B, C, H, W].
    """
    def __init__(self, in_channels=3, base=64, latent_ch=256, out_activation="sigmoid"):
        super().__init__()
        # Encoder: /2 size each down block (×4 => /16 total)
        self.enc = nn.Sequential(
            ConvBlock(in_channels, base),                         # H,W
            ConvBlock(base, base),                                # H,W
            ConvBlock(base, base*2, k=4, s=2, p=1),               # H/2
            ConvBlock(base*2, base*2),
            ConvBlock(base*2, base*4, k=4, s=2, p=1),             # H/4
            ConvBlock(base*4, base*4),
            ConvBlock(base*4, base*8, k=4, s=2, p=1),             # H/8
            ConvBlock(base*8, base*8),
            ConvBlock(base*8, latent_ch, k=4, s=2, p=1),          # H/16
            nn.Flatten(), 
            nn.Linear(latent_ch*8*8, 1024), nn.ReLU(),
            nn.Linear(1024, 16)
            
        )
        # Decoder: ×2 size each up block (×4 => ×16 total)
        self.dec = nn.Sequential(
            nn.Linear(16, 1024), nn.ReLU(),
            nn.Linear(1024, latent_ch*8*8), nn.ReLU(),
            Reshape(latent_ch, 8, 8),
            DeconvBlock(latent_ch, base*8, k=4, s=2, p=1),        # H/8
            ConvBlock(base*8, base*8),
            DeconvBlock(base*8, base*4, k=4, s=2, p=1),           # H/4
            ConvBlock(base*4, base*4),
            DeconvBlock(base*4, base*2, k=4, s=2, p=1),           # H/2
            ConvBlock(base*2, base*2),
            DeconvBlock(base*2, base,   k=4, s=2, p=1),           # H
            ConvBlock(base, base),
            nn.Conv2d(base, in_channels, kernel_size=3, padding=1),
        )
        if out_activation == "sigmoid":
            self.out_act = nn.Sigmoid()   # use if inputs are normalized to [0,1]
        elif out_activation == "tanh":
            self.out_act = nn.Tanh()      # use if inputs are normalized to [-1,1]
        else:
            self.out_act = nn.Identity()

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return self.out_act(x_hat), z