import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv1(x)))

class ConditionalUNet1D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_dim=64):
        """
        in_channels = 2 (1 for Noisy Latent x_t + 1 for Condition Noisy Input)
        out_channels = 1 (Predicted Clean x_0)
        """
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, base_dim),
            nn.SiLU()
        )

        # Downsample
        self.inc = Block(in_channels, base_dim)
        self.down1 = Block(base_dim, base_dim * 2)
        self.down2 = Block(base_dim * 2, base_dim * 4)

        self.pool = nn.MaxPool1d(2)

        # Bottleneck
        self.bot = Block(base_dim * 4, base_dim * 4)

        # Upsample (using Upsample + Conv instead of ConvTranspose for 1D simplicity)
        self.up1 = nn.Upsample(scale_factor=2)
        self.up_conv1 = Block(base_dim * 4 + base_dim * 2, base_dim * 2) # Skip connection
        self.up2 = nn.Upsample(scale_factor=2)
        self.up_conv2 = Block(base_dim * 2 + base_dim, base_dim) # Skip connection

        self.outc = nn.Conv1d(base_dim, out_channels, 1)

    def forward(self, x, t, condition):
        """
        x: Noisy Latent (Batch, 1, Length)
        t: Timestep (Batch,)
        condition: Artifact Noisy Input (Batch, 1, Length)
        """
        # Embed Time
        t_emb = self.time_mlp(t) # [B, dim]
        t_emb = t_emb.unsqueeze(-1) # [B, dim, 1] for broadcasting

        # Concatenate x_t and Condition
        x = torch.cat([x, condition], dim=1) # [B, 2, L]

        # Down
        x1 = self.inc(x)
        x2 = self.pool(self.down1(x1 + t_emb))
        x3 = self.pool(self.down2(x2))

        # Bottleneck
        x_bot = self.bot(x3)

        # Up
        x = self.up1(x_bot)
        # Handle shape mismatch due to pooling odd lengths (187 -> 93 -> 46)
        if x.shape[-1] != x2.shape[-1]:
            x = F.interpolate(x, size=x2.shape[-1])
            
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv1(x)

        x = self.up2(x)
        if x.shape[-1] != x1.shape[-1]:
            x = F.interpolate(x, size=x1.shape[-1])
            
        x = torch.cat([x, x1 + t_emb], dim=1)
        x = self.up_conv2(x)

        return self.outc(x)