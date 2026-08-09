import torch.nn as nn
class EmotionAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    def forward(self, pred_va, target_va):
        return self.l1(pred_va, target_va)
