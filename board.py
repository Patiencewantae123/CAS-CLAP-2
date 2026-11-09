"""Comparison-board helpers shared by app.py (Streamlit) and app_gradio.py (Gradio, Hugging Face Space).

An item is one board column, a dict with:
  id, title, image (bytes), source, gt_audio (Path or None), gt_window ((start, end) s, or None for
  the whole recording), gt_captions (list), and once processed, result (CASVerifier.decide output)
  and wav ({"baseline": Path, "casv": Path}).
"""

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from casv.audio import AUDIOLDM_MODEL_ID, GENERATOR_SAMPLE_RATE, load_audio, save_audio
from casv.verifier import CLIP_MODEL_ID
from scripts.figures import render_page


def open_image(data: bytes) -> Image.Image:
    """RGB image, upright for phone photos with an EXIF rotation."""
    return ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")


def upload_item(name: str, data: bytes, recording: Path | None = None, caption: str = "") -> dict:
    """Board column for an uploaded image (and its optional real recording)."""
    stem = Path(name).stem.replace("_", " ").replace("-", " ").strip()
    return {
        "id": f"upload-{hashlib.sha1(data).hexdigest()[:10]}",
        "title": stem.title() or "Upload",
        "image": data,
        "source": f"uploaded image '{name}'",
        "gt_audio": recording,
        "gt_window": None,  # the whole recording
        "gt_captions": [caption] if caption else [],
    }


def clip_path(clips_dir: Path, prompt: str, settings: dict) -> Path:
    """Cache file of a generated clip: same model, prompt and settings give the same audio."""
    key = json.dumps([AUDIOLDM_MODEL_ID, prompt, settings["seed"], settings["duration"], settings["steps"]])
    return clips_dir / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.wav"


def save_clip(path: Path, audio, prompt: str, settings: dict, seconds: float) -> None:
    """Save a generated clip with a JSON note of what produced it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_audio(path, audio, GENERATOR_SAMPLE_RATE)
    note = {
        "model": AUDIOLDM_MODEL_ID,
        "prompt": prompt,
        **settings,
        "generation_seconds": round(seconds, 1),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    path.with_suffix(".json").write_text(json.dumps(note, indent=2), encoding="utf-8")


def convert_recording(data: bytes, name: str, out_dir: Path) -> Path:
    """A ground-truth recording as 16 kHz mono WAV (the rate of every spectrogram).

    Raises RuntimeError when the file cannot be decoded.
    """
    path = out_dir / f"{hashlib.sha1(data).hexdigest()[:16]}.wav"
    if path.exists():
        return path
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / f"recording{Path(name).suffix.lower()}"
        raw.write_bytes(data)
        audio = load_audio(raw, GENERATOR_SAMPLE_RATE)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_audio(path, audio, GENERATOR_SAMPLE_RATE)
    return path


def export_figure(items: list[dict], run_dir: Path, config_name: str, config, settings: dict, made_in: str) -> Path:
    """Copy each column into run_dir in test.py's layout and render it with scripts/figures.py.

    Returns the figure path; its caption file sits next to it.
    """
    records = []
    for item in items:
        item_dir = run_dir / item["id"]
        item_dir.mkdir(parents=True, exist_ok=True)
        open_image(item["image"]).save(item_dir / "input.png")
        shutil.copyfile(item["wav"]["baseline"], item_dir / "baseline.wav")
        shutil.copyfile(item["wav"]["casv"], item_dir / "casv.wav")
        if item["gt_audio"] is not None:
            shutil.copyfile(item["gt_audio"], item_dir / "gt.wav")
        result = item["result"]
        record = {
            "id": item["id"],
            "label": item["title"],
            "source": item["source"],
            "gt_captions": item["gt_captions"],
            "gt_window": item["gt_window"],
            "top_scene": list(result["top_scenes"][0]),
            "top_object": list(result["top_objects"][0]),
            "env_probs": result["env_probs"],
            "status": result["status"],
            "n_rejected": result["n_rejected"],
            "prompts": {"baseline": result["baseline_prompt"], "casv": result["casv_prompt"]},
            "verdicts": result["verdicts"],
            "config_name": config_name,
            "config": asdict(config),
            "settings": settings,
        }
        (item_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)

    provenance = (
        f"CAS-V comparison built in {made_in} on {datetime.now():%Y-%m-%d}. The images were chosen by hand, "
        f"not sampled: this figure is an illustration, not a result. Existing research: "
        f"{AUDIOLDM_MODEL_ID} prompted with the top-1 retrieved AudioCaps train caption, unverified. Ours: "
        f"the same generator and seed ({settings['seed']}; {settings['duration']} s, {settings['steps']} "
        f"steps) prompted with the CAS-V-verified caption ({config_name} verifier configuration); verified "
        f"concepts and sound type are phrases of that caption. Recognition: {CLIP_MODEL_ID} zero-shot over "
        f"Places365 scenes and ImageNet objects. Target sound: the real recording where one exists (AudioCaps "
        f"examples: 5 s window centred on the input frame; uploads: the whole recording), otherwise marked "
        f"'not provided'. Spectrograms as in scripts/figures.py."
    )
    figure = run_dir / "casv_comparison.png"
    render_page(run_dir, records, figure, provenance)
    return figure
