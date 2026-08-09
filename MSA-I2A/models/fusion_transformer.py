import torch
import torch.nn as nn

class MultimodalFusionTransformer(nn.Module):
    def __init__(self, embed_dim=512, nhead=8, num_layers=4):
        super().__init__()
        self.proj_clip = nn.Linear(768, embed_dim)
        self.proj_sam = nn.Linear(256, embed_dim)
        self.proj_emotion = nn.Linear(128, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, clip_tokens, sam_embed, emotion_embed):
        clip_proj = self.proj_clip(clip_tokens)
        sam_proj = self.proj_sam(sam_embed).unsqueeze(1)
        emo_proj = self.proj_emotion(emotion_embed).unsqueeze(1)
        fused_tokens = torch.cat([clip_proj, sam_proj, emo_proj], dim=1)
        return self.transformer(fused_tokens)
