import torch
import torch.nn as nn

class EmotionEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc_va = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 2), nn.Tanh()
        )
        self.projection = nn.Linear(2, embed_dim)

    def forward(self, image: torch.Tensor):
        x = torch.flatten(self.conv_net(image), 1)
        va_scores = self.fc_va(x)
        return self.projection(va_scores), va_scores
