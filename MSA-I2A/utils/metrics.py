import torch

def compute_fad_score(real_audio_features, gen_audio_features):
    mu1, mu2 = real_audio_features.mean(dim=0), gen_audio_features.mean(dim=0)
    diff = mu1 - mu2
    return float(torch.dot(diff, diff).item())
