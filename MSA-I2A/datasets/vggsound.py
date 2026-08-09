import torch
from torch.utils.data import Dataset

class VGGSoundDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [f"sample_{i}" for i in range(100)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        mask = torch.zeros(1, 224, 224)
        mask[:, 50:150, 50:150] = 1.0
        mel_spectrogram = torch.randn(1, 64, 128)
        va_targets = torch.tensor([0.5, -0.2], dtype=torch.float32)
        return {"image": image, "mask": mask, "mel": mel_spectrogram, "va_targets": va_targets}
