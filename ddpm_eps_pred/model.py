"""
Denoiser eps_theta(x_t, t) that predicts the NOISE eps added to x0 to form
x_t, rather than predicting x0 directly (the noise-prediction / epsilon
parameterization used by Ho et al. 2020's original DDPM).

Small U-Net: two downsampling conv blocks, a bottleneck, two upsampling conv
blocks with skip connections. The timestep is embedded with a sinusoidal
embedding (as in the original DDPM paper) and injected into every conv block
as a per-channel bias (FiLM-style, additive only).
"""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (batch, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (batch, dim)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, out_ch)

    def forward(self, x, t_emb):
        h = self.act(self.norm(self.conv(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        return h


class NoisePredictor(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, time_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        c = base_channels
        self.down1 = ConvBlock(in_channels, c, time_dim)          # 28x28
        self.pool1 = nn.Conv2d(c, c, 3, stride=2, padding=1)      # 28 -> 14
        self.down2 = ConvBlock(c, c * 2, time_dim)                # 14x14
        self.pool2 = nn.Conv2d(c * 2, c * 2, 3, stride=2, padding=1)  # 14 -> 7

        self.bottleneck = ConvBlock(c * 2, c * 2, time_dim)       # 7x7

        self.up2 = nn.ConvTranspose2d(c * 2, c * 2, 4, stride=2, padding=1)  # 7 -> 14
        self.dec2 = ConvBlock(c * 2 + c * 2, c, time_dim)         # concat skip from down2
        self.up1 = nn.ConvTranspose2d(c, c, 4, stride=2, padding=1)         # 14 -> 28
        self.dec1 = ConvBlock(c + c, c, time_dim)                 # concat skip from down1

        self.out_conv = nn.Conv2d(c, in_channels, 1)

    def forward(self, x, t):
        """x: (batch, 1, 28, 28), t: (batch,) 1-indexed timesteps -> eps_hat"""
        t_emb = self.time_embed(t)

        h1 = self.down1(x, t_emb)          # (b, c, 28, 28)
        h = self.pool1(h1)                 # (b, c, 14, 14)
        h2 = self.down2(h, t_emb)          # (b, 2c, 14, 14)
        h = self.pool2(h2)                 # (b, 2c, 7, 7)

        h = self.bottleneck(h, t_emb)      # (b, 2c, 7, 7)

        h = self.up2(h)                    # (b, 2c, 14, 14)
        h = self.dec2(torch.cat([h, h2], dim=1), t_emb)  # (b, c, 14, 14)
        h = self.up1(h)                    # (b, c, 28, 28)
        h = self.dec1(torch.cat([h, h1], dim=1), t_emb)  # (b, c, 28, 28)

        eps_hat = self.out_conv(h)
        return eps_hat
