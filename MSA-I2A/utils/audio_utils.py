import torch
import torchaudio

def save_waveform(waveform: torch.Tensor, path: str, sample_rate: int = 16000):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(path, waveform, sample_rate)
