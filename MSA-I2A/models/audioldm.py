import torch
import torch.nn as nn

class LatentDiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, context_dim=512):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 128))
        self.down1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.cross_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.context_proj = nn.Linear(context_dim, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.out = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x, timestep, context):
        t_emb = self.time_embed(timestep).unsqueeze(-1).unsqueeze(-1)
        h = self.down2(self.down1(x) + t_emb[:, :64])
        B, C, F, T = h.shape
        h_flat = h.view(B, C, F * T).permute(0, 2, 1)
        ctx_proj = self.context_proj(context)
        attn_out, _ = self.cross_attn(h_flat, ctx_proj, ctx_proj)
        h_attn = attn_out.permute(0, 2, 1).view(B, C, F, T)
        return self.out(self.up1(h_attn))
