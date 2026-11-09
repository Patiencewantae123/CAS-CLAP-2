# =============================================================================
# Audio generation (AudioLDM), audio I/O, mel spectrograms and CLAP scoring.
# =============================================================================

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function

AUDIOLDM_MODEL_ID = "cvssp/audioldm-s-full-v2"
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
GENERATOR_SAMPLE_RATE = 16000


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------


def load_audio(path, target_sr):
    """Read an audio file as mono float32, resampled to target_sr."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        g = np.gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    return audio


def save_audio(path, audio, sr):
    """Write mono audio as 16-bit PCM."""
    sf.write(str(path), audio, sr, subtype="PCM_16")


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------


class AudioGenerator:
    """Text-to-audio generation with AudioLDM (diffusers)."""

    def __init__(self, model_id=AUDIOLDM_MODEL_ID, device="cpu"):
        from diffusers import AudioLDMPipeline

        self.model_id = model_id
        self.device = device
        self.pipe = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, prompt, seed, duration=5.0, steps=50, guidance_scale=2.5):
        """Generate one clip for prompt; the same seed gives the same audio."""
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = self.pipe(
            prompt,
            num_inference_steps=steps,
            audio_length_in_s=duration,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return np.asarray(result.audios[0], dtype=np.float32)


# -----------------------------------------------------------------------------
# Mel spectrogram (dB) for figures
# -----------------------------------------------------------------------------


def mel_spectrogram_db(audio, sr=GENERATOR_SAMPLE_RATE, n_fft=1024, hop=160, n_mels=64):
    """Log-mel spectrogram in dB for figures."""
    filters = mel_filter_bank(
        num_frequency_bins=n_fft // 2 + 1,
        num_mel_filters=n_mels,
        min_frequency=0.0,
        max_frequency=sr / 2,
        sampling_rate=sr,
        norm="slaney",
        mel_scale="slaney",
    )
    return spectrogram(
        audio,
        window_function(n_fft, "hann"),
        frame_length=n_fft,
        hop_length=hop,
        power=2.0,
        mel_filters=filters,
        log_mel="dB",
        reference=float(np.max(np.abs(audio)) ** 2 + 1e-10),
        db_range=80.0,
    )


# -----------------------------------------------------------------------------
# CLAP evaluation
# -----------------------------------------------------------------------------


class CLAPScorer:
    """CLAP audio/text embeddings for evaluation (cosine similarity)."""

    sample_rate = 48000

    def __init__(self, model_id=CLAP_MODEL_ID, device="cpu"):
        from transformers import ClapModel, ClapProcessor

        self.model_id = model_id
        self.device = device
        self.model = ClapModel.from_pretrained(model_id).to(device).eval()
        self.processor = ClapProcessor.from_pretrained(model_id)

    @staticmethod
    def _as_tensor(output):
        return output if torch.is_tensor(output) else output.pooler_output

    @torch.no_grad()
    def audio_embedding(self, audio, sr):
        """L2-normalised CLAP embedding of an audio clip."""
        if sr != self.sample_rate:
            g = np.gcd(sr, self.sample_rate)
            audio = resample_poly(audio, self.sample_rate // g, sr // g).astype(np.float32)
        # Over 10 s the processor crops at an unseeded random offset, and a shift of
        # a few ms changes the embedding, so fix the crop to make scores repeatable.
        audio = audio[: self.processor.feature_extractor.nb_max_samples]
        inputs = self.processor(audio=audio, sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)
        emb = self._as_tensor(self.model.get_audio_features(**inputs))
        return torch.nn.functional.normalize(emb, dim=-1)[0]

    @torch.no_grad()
    def text_embedding(self, text):
        """L2-normalised CLAP embedding of one text."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        emb = self._as_tensor(self.model.get_text_features(**inputs))
        return torch.nn.functional.normalize(emb, dim=-1)[0]

    @torch.no_grad()
    def text_embeddings(self, texts, batch_size=64):
        """L2-normalised CLAP embeddings of many texts, batched."""
        chunks = []
        for start in range(0, len(texts), batch_size):
            inputs = self.processor(text=texts[start : start + batch_size], return_tensors="pt", padding=True).to(
                self.device
            )
            emb = self._as_tensor(self.model.get_text_features(**inputs))
            chunks.append(torch.nn.functional.normalize(emb, dim=-1))
        return torch.cat(chunks)

    @staticmethod
    def cosine(a, b):
        """Cosine similarity of two L2-normalised embeddings."""
        return float((a @ b).item())
