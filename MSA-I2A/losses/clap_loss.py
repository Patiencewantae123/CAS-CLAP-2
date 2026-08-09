import torch.nn as nn
import torch.nn.functional as F
class CLAPCosineLoss(nn.Module):
    def forward(self, audio_embed, visual_embed):
        a_norm = F.normalize(audio_embed, p=2, dim=-1)
        v_norm = F.normalize(visual_embed, p=2, dim=-1)
        return (1.0 - (a_norm * v_norm).sum(dim=-1)).mean()
