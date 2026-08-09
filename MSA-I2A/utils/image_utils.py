import torch
import numpy as np

def extract_color_histogram(image_np):
    hist, _ = np.histogram(image_np, bins=32, range=(0, 255))
    return torch.tensor(hist, dtype=torch.float32) / hist.sum()
