import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class VisualCLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
        return outputs.last_hidden_state
