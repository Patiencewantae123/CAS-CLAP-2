import torch.nn as nn
class DiffusionMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred_noise, target_noise):
        return self.mse(pred_noise, target_noise)
