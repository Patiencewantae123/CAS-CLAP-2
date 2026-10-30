import torch
from models.msa_i2a import MSAI2AModel
from utils.audio_utils import save_waveform

def run_inference(image_path: str, output_audio_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSAI2AModel().to(device)
    model.eval()

    image_tensor = torch.randn(1, 3, 224, 224).to(device)
    mask_tensor = torch.ones(1, 1, 224, 224).to(device)

    print(f"Processing image: {image_path}...")
    with torch.no_grad():
        waveform = model.generate_audio(image_tensor, mask_tensor)

    save_waveform(waveform.squeeze(0).cpu(), output_audio_path, sample_rate=16000)
    print(f"Generated soundscape successfully saved to: {output_audio_path}")

if __name__ == "__main__":
    run_inference("test_scene.jpg", "output_soundscape.wav")
