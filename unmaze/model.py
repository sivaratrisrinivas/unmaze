"""The Denoiser: a small conditional U-Net that predicts the clean Path Mask."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from unmaze.diffusion import TRAIN_TIMESTEPS


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(8, channels), channels)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, temb: int):
        super().__init__()
        self.norm1, self.conv1 = _norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2, self.conv2 = _norm(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.temb = nn.Linear(temb, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Lets every pixel look at every other, so the Denoiser can follow a corridor across the maze."""

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = _norm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class UNet(nn.Module):
    """Predicts the clean Path Mask from a noisy one, the Puzzle's channels and the noise level.

    The Grid is padded with wall up to a multiple of 4 and cropped back, so any maze size works.
    """

    def __init__(self, base: int = 48):
        super().__init__()
        self.base = base
        emb = 4 * base
        self.time_mlp = nn.Sequential(nn.Linear(base, emb), nn.SiLU(), nn.Linear(emb, emb))
        c1, c2, c3 = base, 2 * base, 4 * base

        self.stem = nn.Conv2d(1 + 3 + 2, c1, 3, padding=1)  # noisy mask + puzzle + (row, col) pixel positions
        self.down1 = nn.ModuleList([ResBlock(c1, c1, emb), ResBlock(c1, c1, emb)])
        self.pool1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.down2 = ResBlock(c2, c2, emb)
        self.attn2 = SelfAttention(c2)
        self.pool2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)

        self.mid1, self.mid_attn, self.mid2 = ResBlock(c3, c3, emb), SelfAttention(c3), ResBlock(c3, c3, emb)

        self.unpool2 = nn.Conv2d(c3, c2, 3, padding=1)
        self.up2, self.up_attn2 = ResBlock(2 * c2, c2, emb), SelfAttention(c2)
        self.unpool1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.up1 = ResBlock(2 * c1, c1, emb)
        self.out = nn.Sequential(_norm(c1), nn.SiLU(), nn.Conv2d(c1, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.base // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
        angles = (t.float() / TRAIN_TIMESTEPS * 1000)[:, None] * freqs[None]
        return self.time_mlp(torch.cat([angles.sin(), angles.cos()], dim=1))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h, w = x_t.shape[-2:]
        pad = (0, -w % 4, 0, -h % 4)
        # Pad each channel with what the new pixels mean: no path, wall, and no start or goal.
        x_t = F.pad(x_t, pad, value=-1.0)
        walls, marks = F.pad(cond[:, :1], pad, value=1.0), F.pad(cond[:, 1:], pad, value=0.0)
        b, _, ph, pw = x_t.shape
        rows = (torch.arange(ph, device=x_t.device) / 8).view(1, 1, ph, 1).expand(b, 1, ph, pw)
        cols = (torch.arange(pw, device=x_t.device) / 8).view(1, 1, 1, pw).expand(b, 1, ph, pw)
        x = torch.cat([x_t, walls, marks, rows, cols], dim=1)

        emb = self._time_embedding(t)
        x = self.stem(x)
        for block in self.down1:
            x = block(x, emb)
        skip1 = x
        x = self.attn2(self.down2(self.pool1(x), emb))
        skip2 = x
        x = self.mid2(self.mid_attn(self.mid1(self.pool2(x), emb)), emb)

        x = self.unpool2(F.interpolate(x, scale_factor=2))
        x = self.up_attn2(self.up2(torch.cat([x, skip2], dim=1), emb))
        x = self.unpool1(F.interpolate(x, scale_factor=2))
        x = self.up1(torch.cat([x, skip1], dim=1), emb)
        return self.out(x)[:, :, :h, :w]
