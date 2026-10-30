import torch
import torch.nn as nn

class HiFiGANVocoder(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=7, padding=3), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=8, padding=4), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=16, stride=8, padding=4), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, kernel_size=7, padding=3), nn.Tanh()
        )

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        return self.generator(mel_spectrogram.squeeze(1))
