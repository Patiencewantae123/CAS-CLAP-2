"""Comparison-board helpers shared by app.py and app_gradio.py (no models needed)."""

import io

import matplotlib
import numpy as np
import pytest
import soundfile as sf
from PIL import Image

import board
from casv.audio import GENERATOR_SAMPLE_RATE
from casv.verifier import VerifierConfig

matplotlib.use("Agg")
SETTINGS = {"seed": 42, "duration": 5.0, "steps": 50}


def png_bytes(colour="red"):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def wav_bytes(sr, seconds, channels=1):
    buffer = io.BytesIO()
    tone = 0.1 * np.sin(2 * np.pi * 440 * np.arange(int(sr * seconds)) / sr)
    sf.write(buffer, np.stack([tone] * channels, axis=1), sr, format="WAV")
    return buffer.getvalue()


def test_clip_cache_key_follows_prompt_and_settings(tmp_path):
    path = board.clip_path(tmp_path, "a dog barks", SETTINGS)
    assert path == board.clip_path(tmp_path, "a dog barks", dict(SETTINGS))
    assert path != board.clip_path(tmp_path, "a dog barks", {**SETTINGS, "seed": 7})
    assert path != board.clip_path(tmp_path, "a cat meows", SETTINGS)


def test_upload_item_names_and_dedupes_by_content():
    item = board.upload_item("busy_road-2.jpg", png_bytes())
    assert item["title"] == "Busy Road 2"
    assert item["id"] == board.upload_item("other_name.png", png_bytes())["id"]
    assert item["gt_audio"] is None and item["gt_window"] is None and item["gt_captions"] == []


def test_recording_is_converted_to_16k_mono_and_bad_audio_is_refused(tmp_path):
    path = board.convert_recording(wav_bytes(44100, 0.5, channels=2), "beach.wav", tmp_path)
    audio, sr = sf.read(path)
    assert sr == GENERATOR_SAMPLE_RATE and audio.ndim == 1 and len(audio) == GENERATOR_SAMPLE_RATE // 2
    with pytest.raises(RuntimeError):
        board.convert_recording(b"not audio at all", "broken.wav", tmp_path)


def test_export_figure_renders_board_with_honest_caption(tmp_path):
    clips = tmp_path / "clips"
    item = board.upload_item("beach.png", png_bytes("blue"))
    item["wav"] = {}
    for arm in ("baseline", "casv"):
        item["wav"][arm] = board.clip_path(clips, f"{arm} prompt", SETTINGS)
        board.save_clip(item["wav"][arm], np.zeros(GENERATOR_SAMPLE_RATE, np.float32), f"{arm} prompt", SETTINGS, 1.0)
    item["result"] = {
        "top_scenes": [("beach", 0.6)],
        "top_objects": [("seashore", 0.4)],
        "env_probs": {"indoor": 0.1, "outdoor": 0.9},
        "status": "ACCEPT",
        "n_rejected": 3,
        "baseline_prompt": "baseline prompt",
        "casv_prompt": "Waves crash on a shore",
        "verdicts": [],
    }

    figure = board.export_figure([item], tmp_path / "run", "Default (paper)", VerifierConfig(), SETTINGS, "a test")

    assert figure.exists() and Image.open(figure).width > 300
    caption = figure.with_name("casv_comparison_caption.txt").read_text(encoding="utf-8")
    assert "illustration, not a result" in caption
    assert "uploaded image 'beach.png'" in caption and "AudioCaps upload-" not in caption
    assert not (figure.parent / item["id"] / "gt.wav").exists()  # no recording: the panel says 'not provided'
