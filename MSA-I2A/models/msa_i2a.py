import torch
import torch.nn as nn
from .clip_encoder import VisualCLIPEncoder
from .emotion_encoder import EmotionEncoder
from .sam_encoder import ObjectSAMEncoder
from .fusion_transformer import MultimodalFusionTransformer
from .audioldm import LatentDiffusionUNet
from .hifigan import HiFiGANVocoder

class MSAI2AModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_encoder = VisualCLIPEncoder()
        self.emotion_encoder = EmotionEncoder(embed_dim=128)
        self.sam_encoder = ObjectSAMEncoder(out_dim=256)
        self.fusion = MultimodalFusionTransformer(embed_dim=512)
        self.diffusion_unet = LatentDiffusionUNet(in_channels=1, context_dim=512)
        self.vocoder = HiFiGANVocoder(in_channels=64)

    def forward(self, image, mask, noisy_mel, t):
        clip_feats = self.clip_encoder(image)
        sam_feats = self.sam_encoder(mask)
        emotion_feats, va_scores = self.emotion_encoder(image)
        context = self.fusion(clip_feats, sam_feats, emotion_feats)
        predicted_noise = self.diffusion_unet(noisy_mel, t, context)
        return predicted_noise, va_scores

    @torch.no_grad()
    def generate_audio(self, image, mask):
        self.eval()
        B = image.size(0)
        clip_feats = self.clip_encoder(image)
        sam_feats = self.sam_encoder(mask)
        emotion_feats, _ = self.emotion_encoder(image)
        context = self.fusion(clip_feats, sam_feats, emotion_feats)
        mel_latent = torch.randn((B, 1, 64, 128), device=image.device)
        for step in reversed(range(10)):
            t = torch.full((B, 1), step / 10.0, device=image.device)
            mel_latent -= 0.1 * self.diffusion_unet(mel_latent, t, context)
        return self.vocoder(mel_latent)
