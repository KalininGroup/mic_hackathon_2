"""Tiny UNet-based flow model (latest notebook version)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResBlock", "Down", "Up", "TinyUNetFlow"]


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_ch: int):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.tproj = nn.Linear(t_ch, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.conv1(x))
        h = h + self.tproj(t_emb)[:, :, None, None]
        h = F.silu(self.conv2(h))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = self.pool(x)
        return F.silu(self.conv(x))


class Up(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return F.silu(self.conv(x))


class TinyUNetFlow(nn.Module):
    """
    Input: concat([x_t, x0]) -> 2 channels
    Output: velocity v(x_t, x0, t) -> 1 channel
    """

    def __init__(self, base: int = 64, t_ch: int = 128):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(1, t_ch),
            nn.SiLU(),
            nn.Linear(t_ch, t_ch),
        )

        self.in_conv = nn.Conv2d(2, base, 3, padding=1)

        self.rb1 = ResBlock(base, base, t_ch)
        self.down1 = Down(base)

        self.rb2 = ResBlock(base, base * 2, t_ch)
        self.down2 = Down(base * 2)

        self.mid = ResBlock(base * 2, base * 2, t_ch)

        self.up2 = Up(base * 2)
        self.rb_up2 = ResBlock(base * 2 + base * 2, base * 2, t_ch)

        self.up1 = Up(base * 2)
        self.rb_up1 = ResBlock(base * 2 + base, base, t_ch)

        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x_t, x0, t):
        # t: [B] in [0,1]
        t_emb = self.t_mlp(t[:, None])

        h0 = F.silu(self.in_conv(torch.cat([x_t, x0], dim=1)))  # [B,base,H,W]

        h1 = self.rb1(h0, t_emb)         # [B,base,H,W]
        d1 = self.down1(h1)              # [B,base,H/2,W/2]

        h2 = self.rb2(d1, t_emb)         # [B,2base,H/2,W/2]
        d2 = self.down2(h2)              # [B,2base,H/4,W/4]

        m = self.mid(d2, t_emb)          # [B,2base,H/4,W/4]

        u2 = self.up2(m)                 # [B,2base,H/2,W/2]
        u2 = torch.cat([u2, h2], dim=1)  # skip
        u2 = self.rb_up2(u2, t_emb)      # [B,2base,H/2,W/2]

        u1 = self.up1(u2)                # [B,2base,H,W]
        u1 = torch.cat([u1, h1], dim=1)  # skip
        u1 = self.rb_up1(u1, t_emb)      # [B,base,H,W]

        return self.out_conv(u1)
