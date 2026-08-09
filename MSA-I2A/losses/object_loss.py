import torch.nn as nn
class ObjectFoleyMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, mel_energy, mask_density):
        return self.bce(mel_energy, mask_density)
