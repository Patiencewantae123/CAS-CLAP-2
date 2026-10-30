import torch
import torch.nn as nn

class ObjectSAMEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)), nn.Flatten(),
            nn.Linear(32 * 8 * 8, out_dim)
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.mask_conv(mask)
