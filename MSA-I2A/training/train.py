import torch
from torch.utils.data import DataLoader
from models.msa_i2a import MSAI2AModel
from datasets.vggsound import VGGSoundDataset
from losses.diffusion_loss import DiffusionMSELoss
from losses.emotion_loss import EmotionAlignmentLoss

def train_one_epoch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VGGSoundDataset(root_dir="./data")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = MSAI2AModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    diff_criterion, emo_criterion = DiffusionMSELoss(), EmotionAlignmentLoss()

    model.train()
    print("Beginning MSA-I2A Training Loop...")
    for batch_idx, batch in enumerate(loader):
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        mels, va_targets = batch["mel"].to(device), batch["va_targets"].to(device)
        
        noise = torch.randn_like(mels).to(device)
        timesteps = torch.rand((mels.size(0), 1), device=device)
        noisy_mels = mels + noise * timesteps.unsqueeze(-1)

        optimizer.zero_grad()
        pred_noise, pred_va = model(images, masks, noisy_mels, timesteps)
        loss = diff_criterion(pred_noise, noise) + 0.1 * emo_criterion(pred_va, va_targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}/{len(loader)} | Total Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_one_epoch()
