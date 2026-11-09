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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image, ImageOps

from casv.audio import AUDIOLDM_MODEL_ID, GENERATOR_SAMPLE_RATE, load_audio, save_audio
from casv.verifier import CLIP_MODEL_ID
from scripts.figures import GT_WINDOW, render_page

# Paper evaluation graphs (journal_results/graphs/), shown by both apps. The graphs print their own numbers,
# so the captions describe them without repeating figures that a re-run would change.
PAPER_RESULTS_INTRO = (
    "These graphs come from the paper's evaluation on AudioCaps, not from the images on this page. In short: "
    "CAS-V changes the prompt for more than half of the 400 test images, but by CLAP similarity the generated "
    "audio is not measurably closer to the real recordings. A blind listening test is in preparation."
)
PAPER_GRAPHS = [
    (
        "fig3_generated_audio_clap.png",
        "Generated audio",
        "AudioLDM audio from both prompts, with the same seed, compared with the human captions and the real "
        "recording of 400 test clips (means with 95% confidence intervals; Wilcoxon p in the titles). The "
        "difference is small and not significant. Right: each clip where CAS-V changed the prompt.",
    ),
    (
        "fig2_prompt_selection.png",
        "Prompt selection",
        "CLAP similarity between the chosen text prompt and the real recording: the unverified top-1 caption "
        "(baseline), CAS-V with its default settings (the paper's method), CAS-V tuned on the validation split "
        "(an ablation) and the best of the 20 candidates (oracle, an upper bound). Tuning helps on validation, "
        "not on the 400 test clips.",
    ),
    (
        "fig1_scene_recognition.png",
        "Scene recognition",
        "How well CLIP recognises the scene on 3,650 Places365 validation images: scene and indoor/outdoor "
        "accuracy (left) and how well its confidence matches its accuracy (right). CAS-V raises an indoor/outdoor "
        "conflict only when that call is confident (p ≥ 0.7).",
    ),
    (
        "fig4_verification_statistics.png",
        "Verification statistics",
        "What the verifier did on the 400 test images: its decisions on all 8,000 candidate captions, the most "
        "frequent explicit conflicts, and the final status per image.",
    ),
]

# Chart tokens (reference data-viz palette). Decisions are states, so they take the status colours, each
# paired with a glyph so colour never carries the meaning alone; recognition is one series (slot-1 blue).
DECISION_STYLE = {
    "ACCEPT": ("#0ca30c", "✓", "accept"),
    "REPAIR": ("#fab219", "~", "repair"),
    "REJECT": ("#d03b3b", "✕", "reject"),
}
SERIES_BLUE = "#2a78d6"
SURFACE, INK, INK_2, MUTED, GRIDLINE, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CHART_FONT = {"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]}
GLYPH_FONT = "DejaVu Sans"  # has ✓ ✕ ←


def open_image(data: bytes) -> Image.Image:
    """RGB image, upright for phone photos with an EXIF rotation."""
    return ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")


def load_examples(examples_dir: Path) -> list[dict]:
    """Figure 5 examples (AudioCaps test clips with real recordings), if the data is present."""
    manifest = examples_dir / "manifest.json"
    return json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []


def example_item(entry: dict, examples_dir: Path) -> dict:
    """Board column for a Figure 5 example: its frame, real recording and human captions."""
    return {
        "id": entry["id"],
        "title": entry["label"].title(),
        "image": (examples_dir / entry["image"]).read_bytes(),
        "source": f"AudioCaps {entry['id']}",
        "gt_audio": examples_dir / entry["gt_audio"],
        "gt_window": GT_WINDOW,  # 5 s of the 10 s clip, centred on the frame, as in the paper figures
        "gt_captions": entry["captions"],
    }


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


def _short(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def _plain_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8.5, length=0)


def _probability_bars(ax, pairs: list, title: str, subtitle: str = "") -> None:
    """Top-5 recognition probabilities: one series, value at each bar tip, no value axis."""
    _plain_axis(ax)
    names = [_short(name.replace("_", " "), 26) for name, _ in pairs]
    probs = [p for _, p in pairs]
    y = np.arange(len(pairs))[::-1]
    ax.barh(y, probs, height=0.6, color=SERIES_BLUE, linewidth=0)
    for yi, p in zip(y, probs, strict=True):
        ax.text(p + max(probs) * 0.03, yi, f"{p:.0%}" if p >= 0.005 else "<1%", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, names)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, max(probs) * 1.3)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=16 if subtitle else 6)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK_2)


def verification_chart(result: dict, config) -> bytes:
    """PNG of one image's verification: the retrieved candidates by score, with their decisions, the thresholds,
    the baseline and the CAS-V choice; beside them the recognised scenes and objects."""
    verdicts = sorted(result["verdicts"], key=lambda v: v["score"], reverse=True)
    chosen, baseline = result["casv_prompt"], result["baseline_prompt"]
    n = len(verdicts)
    with plt.rc_context(CHART_FONT):
        fig = plt.figure(figsize=(12.5, 6.9), dpi=110, facecolor=SURFACE)
        grid = fig.add_gridspec(
            2, 2, width_ratios=[2.3, 1], left=0.3, right=0.98, top=0.8, bottom=0.08, wspace=0.45, hspace=0.6
        )
        ax = fig.add_subplot(grid[:, 0])
        _plain_axis(ax)
        y = np.arange(n)[::-1]  # highest score at the top
        for yi, verdict in zip(y, verdicts, strict=True):
            colour, glyph, _ = DECISION_STYLE[verdict["decision"]]
            ax.barh(yi, verdict["score"], height=0.62, color=colour, linewidth=0)
            roles = [
                name
                for name, prompt in (("CAS-V choice", chosen), ("baseline", baseline))
                if verdict["caption"] == prompt
            ]
            note = (
                glyph + (" conflict" if verdict["conflicts"] else "") + (f"   ← {' and '.join(roles)}" if roles else "")
            )
            ax.text(
                verdict["score"] + 0.012,
                yi,
                note,
                va="center",
                fontsize=8.5,
                fontfamily=GLYPH_FONT,
                color=INK if roles else INK_2,
                fontweight="bold" if roles else "normal",
            )
        ax.set_yticks(y, [_short(v["caption"], 58) for v in verdicts])
        for label, verdict in zip(ax.get_yticklabels(), verdicts, strict=True):
            if verdict["caption"] == chosen:
                label.set_fontweight("bold")
                label.set_color(INK)
        for x, name in ((config.repair_threshold, "repair"), (config.accept_threshold, "accept")):
            ax.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
            ax.text(x, n - 0.2, f"{name} ≥ {x:.2f}", ha="center", va="bottom", fontsize=8, color=MUTED)
        ax.set_ylim(-0.7, n + 0.6)
        ax.set_xlim(0, 1.32)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("verification score", fontsize=9, color=INK_2)

        weights = f"{config.w_visual:.1f}·visual + {config.w_scene:.1f}·scene + {config.w_object:.1f}·object support"
        fig.text(0.02, 0.955, "Why CAS-V chose its caption", fontsize=13, color=INK, fontweight="bold")
        subtitle = f"The {n} AudioCaps captions retrieved for this image, by verification score ({weights}). "
        subtitle += "A conflict with the image rejects a caption whatever its score."
        if chosen not in {v["caption"] for v in verdicts}:
            subtitle += f'\nEvery candidate conflicted, so CAS-V built its own prompt: "{_short(chosen, 90)}".'
        fig.text(0.02, 0.925, subtitle, fontsize=9, color=INK_2, va="top")
        handles = [Patch(color=colour, label=f"{glyph} {word}") for colour, glyph, word in DECISION_STYLE.values()]
        fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.3, 0.86),
            ncol=3,
            frameon=False,
            fontsize=8.5,
            labelcolor=INK_2,
            prop={"family": GLYPH_FONT, "size": 8.5},
            handlelength=1.2,
        )

        env = result["env_probs"]
        _probability_bars(
            fig.add_subplot(grid[0, 1]),
            result["top_scenes"],
            "Recognised scene (Places365)",
            f"indoor {env['indoor']:.0%} · outdoor {env['outdoor']:.0%}",
        )
        _probability_bars(fig.add_subplot(grid[1, 1]), result["top_objects"], "Recognised object (ImageNet)")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", facecolor=SURFACE)
        plt.close(fig)
    return buffer.getvalue()


# Listening level. Real recordings vary by 30 dB (an office keyboard at -40 dBFS RMS, a beach at -12), the
# generated clips sit around -13 to -22, so at one volume setting a quiet recording sounds silent and a louder
# clip sounds "better". Players therefore get loudness-matched copies; figures, metrics and downloads keep
# the original files.
LISTEN_RMS_DB = -23.0
LISTEN_CEILING = 0.9  # about -1 dBFS
LISTEN_NOTE = (
    "All players are at the same loudness (-23 dBFS, peaks limited) so you can compare them by ear; real "
    "recordings play the 5 s around the frame. Spectrograms, figures and downloads use the original files."
)


def _rms_db(audio: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-12))


def _limit(audio: np.ndarray, sr: int) -> np.ndarray:
    """Look-ahead limiter keeping every peak under LISTEN_CEILING."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    # Gain each sample needs, held over +-10 ms and averaged over 10 ms: the average only covers samples whose
    # hold window includes this one, so it never exceeds the gain a peak needs.
    need = np.minimum(1.0, LISTEN_CEILING / np.maximum(np.abs(audio), 1e-12))
    hold = max(1, int(0.01 * sr))
    gain = uniform_filter1d(minimum_filter1d(need, size=2 * hold + 1), size=hold)
    return np.clip(audio * gain, -LISTEN_CEILING, LISTEN_CEILING)


def match_loudness(audio: np.ndarray, sr: int, target_db: float = LISTEN_RMS_DB, max_boost_db: float = 40.0):
    """Gain to target_db RMS through the limiter. Limiting loud peaks lowers the RMS, so the gain is raised
    again until the result is within 0.5 dB of the target (or the boost reaches max_boost_db)."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0 or _rms_db(audio) < -120:  # silence stays silence
        return audio.astype(np.float32)
    gain_db = target_db - _rms_db(audio)
    for _ in range(6):
        gain_db = min(gain_db, max_boost_db)
        out = _limit(audio * 10 ** (gain_db / 20), sr)
        shortfall = target_db - _rms_db(out)
        if shortfall < 0.5 or gain_db >= max_boost_db:
            break
        gain_db += shortfall
    return out.astype(np.float32)


def listening_copy(path: Path, out_dir: Path, window=None) -> Path:
    """A loudness-matched copy of a clip for the players (cut to window=(start_s, end_s) if given)."""
    key = hashlib.sha1(path.read_bytes() + repr(tuple(window) if window else None).encode()).hexdigest()[:16]
    out = out_dir / f"{key}.wav"
    if not out.exists():
        audio = load_audio(path, GENERATOR_SAMPLE_RATE)
        if window:
            audio = audio[int(window[0] * GENERATOR_SAMPLE_RATE) : int(window[1] * GENERATOR_SAMPLE_RATE)]
        out_dir.mkdir(parents=True, exist_ok=True)
        save_audio(out, match_loudness(audio, GENERATOR_SAMPLE_RATE), GENERATOR_SAMPLE_RATE)
    return out


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
        (item_dir / "verification.png").write_bytes(verification_chart(result, config))
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
