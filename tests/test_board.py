"""Comparison-board helpers shared by app.py and app_gradio.py (no models needed)."""

import io
from pathlib import Path

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
    item["result"] = fake_result("baseline prompt", "Waves crash on a shore")

    figure = board.export_figure([item], tmp_path / "run", "Default (paper)", VerifierConfig(), SETTINGS, "a test")

    assert figure.exists() and Image.open(figure).width > 300
    caption = figure.with_name("casv_comparison_caption.txt").read_text(encoding="utf-8")
    assert "illustration, not a result" in caption
    assert "uploaded image 'beach.png'" in caption and "AudioCaps upload-" not in caption
    assert not (figure.parent / item["id"] / "gt.wav").exists()  # no recording: the panel says 'not provided'
    assert Image.open(figure.parent / item["id"] / "verification.png").width > 1000  # the why-this-caption graph
    audio = sorted(p.name for p in (figure.parent / "audio").iterdir())
    assert audio == ["01 Beach - existing research (without CAS-V).wav", "01 Beach - ours (CAS-V).wav", "prompts.txt"]
    assert '"Waves crash on a shore"' in (figure.parent / "audio" / "prompts.txt").read_text(encoding="utf-8")


def verdict(caption, score, decision, conflicts=()):
    return {"caption": caption, "score": score, "decision": decision, "conflicts": list(conflicts)}


def fake_result(baseline, chosen, verdicts=None):
    return {
        "top_scenes": [("beach", 0.6), ("coast", 0.2)],
        "top_objects": [("seashore", 0.4), ("sandbar", 0.1)],
        "env_probs": {"indoor": 0.1, "outdoor": 0.9},
        "status": "ACCEPT",
        "n_rejected": 1,
        "baseline_prompt": baseline,
        "casv_prompt": chosen,
        "verdicts": verdicts
        or [
            verdict(baseline, 0.55, "REPAIR"),
            verdict(chosen, 0.8, "ACCEPT"),
            verdict("A cat meows indoors", 0.7, "REJECT", ["indoor sound, but image is outdoor (p=0.90)"]),
        ],
    }


def test_verification_chart_renders_every_decision_and_the_fallback_prompt():
    for result in (
        fake_result("baseline prompt", "Waves crash on a shore"),
        # every candidate conflicted, so CAS-V built its own prompt from the scene and object
        fake_result(
            "A cat meows",
            "ambient sounds of a beach with a seashore",
            [verdict("A cat meows", 0.9, "REJECT", ["names 'a cat', not visible in image (z=-1.20)"])],
        ),
    ):
        png = board.verification_chart(result, VerifierConfig())
        assert png[:8] == b"\x89PNG\r\n\x1a\n" and Image.open(io.BytesIO(png)).size[0] > 1000


def test_paper_graphs_named_in_the_apps_exist():
    folder = Path(__file__).resolve().parent.parent / "journal_results" / "graphs"
    assert [name for name, _, _ in board.PAPER_GRAPHS if not (folder / name).exists()] == []


def rms_db(audio):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(audio, np.float64) ** 2)))


def test_quiet_recording_is_brought_to_the_listening_level_without_clipping():
    sr = GENERATOR_SAMPLE_RATE
    rng = np.random.default_rng(0)
    # quiet room tone with sharp clicks (like the office keyboard recording: -40 dBFS RMS, high peaks)
    audio = 0.005 * rng.standard_normal(sr * 2)
    audio[:: sr // 8] = 0.3
    out = board.match_loudness(audio, sr)
    assert abs(rms_db(out) - board.LISTEN_RMS_DB) < 1.5
    assert np.abs(out).max() <= board.LISTEN_CEILING + 1e-6
    assert not np.any(board.match_loudness(np.zeros(sr), sr))  # silence stays silence


def test_listening_copy_is_cut_to_the_window_and_leaves_the_original(tmp_path):
    original = tmp_path / "gt.wav"
    original.write_bytes(wav_bytes(GENERATOR_SAMPLE_RATE, 10.0))
    before = original.read_bytes()
    copy = board.listening_copy(original, tmp_path / "listen", (2.5, 7.5))
    audio, sr = sf.read(copy)
    assert len(audio) == 5 * sr and abs(rms_db(audio) - board.LISTEN_RMS_DB) < 0.5
    assert original.read_bytes() == before
    assert board.listening_copy(original, tmp_path / "listen", (2.5, 7.5)) == copy  # cached
