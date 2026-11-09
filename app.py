# =============================================================================
# CAS-V comparison board (Streamlit), laid out like docs/target_figure.png.
#
#   Columns  one per image: Figure 5 examples (with their real recordings)
#            and/or your own uploads (each with an optional real recording)
#   Rows     Input Image | Existing Research (Without CAS-V) |
#            Ours (CAS-V): Semantic Text (Verified Concepts),
#                          Spectrogram (Generated Audio), Caption (Sound Type) |
#            Target Sound (Ground Truth)
#
# Recognition and verification are the casv/ library used for the reported
# results; both prompts go to AudioLDM with the same seed. Nothing is simulated:
# a panel without real data says so. Generated clips are cached by prompt and
# settings in app_data/runs/clips/; "Build figure" renders the board with
# scripts/figures.py into app_data/runs/figure_<timestamp>/.
#
# Usage:
#   py -m streamlit run app.py
# =============================================================================

import hashlib
import html
import json
import shutil
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageOps

from casv.audio import AUDIOLDM_MODEL_ID, GENERATOR_SAMPLE_RATE, load_audio, save_audio
from casv.text import sound_type, verified_concepts
from casv.verifier import CLIP_MODEL_ID, CASVerifier, CLIPRecognizer, VerifierConfig
from scripts.figures import GT_WINDOW, draw_spectrogram, render_page

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "app_data" / "runs"
CLIPS_DIR = RUNS_DIR / "clips"
RECORDINGS_DIR = RUNS_DIR / "recordings"
EXAMPLES_DIR = ROOT / "data" / "audiocaps" / "pairs" / "categories"
TUNED_CONFIG = ROOT / "results" / "tuning_test400" / "casv_tuned_config.json"
MAX_COLUMNS = 7  # as in the target figure
MIN_GRID_UNITS = 4  # fewer images do not stretch to the full page width
RED, BLUE = "#d9483b", "#4a72d6"  # row labels; readable on light and dark themes
STATUS_COLOURS = {"ACCEPT": "#1baf7a", "REPAIR": "#eda100", "REJECT": "#e34948", "UNCERTAIN": "#6250d6"}

st.set_page_config(page_title="CAS-V comparison board", page_icon="🔊", layout="wide")
st.html(
    f"""<style>
    .st-key-ours {{ border: 1.5px solid {BLUE}; border-radius: 0.7rem; padding: 0.6rem;
                    margin: 0.2rem -0.6rem; background: rgba(74, 114, 214, 0.05); }}
    .casv-label {{ text-align: center; font-weight: 700; line-height: 1.25; font-size: 0.92rem; }}
    .casv-label small {{ display: block; font-weight: 600; font-size: 0.78rem; }}
    .casv-label .ours {{ display: block; font-size: 1.05rem; margin-bottom: 0.6rem; }}
    .casv-title {{ text-align: center; font-weight: 700; white-space: nowrap; overflow: hidden;
                   text-overflow: ellipsis; margin-bottom: 0.25rem; }}
    .casv-box {{ border-radius: 6px; padding: 0.35rem 0.45rem; text-align: center; color: #1f3a93;
                 font-size: 0.8rem; line-height: 1.3; display: flex; align-items: center;
                 justify-content: center; }}
    .casv-concepts {{ background: #edf5ea; border: 1px solid #8fbf7f; min-height: 5.4em; }}
    .casv-type {{ background: #eaf0fb; border: 1px solid #8aa6d6; min-height: 2.9em; font-weight: 600; }}
    .casv-empty {{ aspect-ratio: 2.4; border-radius: 4px; background: rgba(128, 128, 128, 0.13);
                   color: rgba(128, 128, 128, 0.95); font-size: 0.75rem; display: flex; align-items: center;
                   justify-content: center; text-align: center; padding: 0 0.5rem; }}
    .casv-note {{ font-size: 0.72rem; opacity: 0.8; line-height: 1.3; margin-top: 0.1rem; }}
    .casv-dot {{ font-size: 0.9em; }}
    </style>"""
)


# -----------------------------------------------------------------------------
# Models, images and audio
# -----------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading CLIP and vocabularies (first run only)...")
def load_verifier() -> CASVerifier:
    """CLIP recogniser and vocabularies, loaded once per server."""
    return CASVerifier(CLIPRecognizer())


@st.cache_resource(show_spinner="Loading AudioLDM (first generation only)...")
def load_generator():
    """AudioLDM pipeline, loaded on the first generation request."""
    from casv.audio import AudioGenerator

    return AudioGenerator()


def open_image(image_bytes: bytes) -> Image.Image:
    """RGB image, upright for phone photos with an EXIF rotation."""
    return ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")


@st.cache_data(show_spinner=False, max_entries=64)
def measure_image(image_bytes: bytes, _verifier: CASVerifier) -> dict:
    """Config-independent recognition and candidate scores, cached by image content."""
    return _verifier.measure(open_image(image_bytes))


@st.cache_data(show_spinner=False, max_entries=64)
def thumbnail(image_bytes: bytes) -> bytes:
    """4:3 centre crop so every column's input image has the same height."""
    buffer = BytesIO()
    ImageOps.fit(open_image(image_bytes), (480, 360)).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


@st.cache_data(show_spinner=False, max_entries=256)
def spectrogram_png(path: str, mtime: float, window: tuple | None) -> bytes:
    """Panel in the paper-figure style (scripts/figures.py); mtime invalidates the cache."""
    fig = plt.figure(figsize=(3.2, 1.33), dpi=150)
    draw_spectrogram(fig.add_axes((0, 0, 1, 1)), Path(path), window)
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def clip_path(prompt: str, settings: dict) -> Path:
    """Cache file of a generated clip: same model, prompt and settings give the same audio."""
    key = json.dumps([AUDIOLDM_MODEL_ID, prompt, settings["seed"], settings["duration"], settings["steps"]])
    return CLIPS_DIR / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.wav"


def generate_clip(generator, prompt: str, settings: dict, path: Path) -> None:
    """Generate one clip and save it with a JSON note of what produced it."""
    start = time.time()
    audio = generator.generate(prompt, seed=settings["seed"], duration=settings["duration"], steps=settings["steps"])
    path.parent.mkdir(parents=True, exist_ok=True)
    save_audio(path, audio, GENERATOR_SAMPLE_RATE)
    note = {
        "model": AUDIOLDM_MODEL_ID,
        "prompt": prompt,
        **settings,
        "generation_seconds": round(time.time() - start, 1),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    path.with_suffix(".json").write_text(json.dumps(note, indent=2), encoding="utf-8")


def store_recording(upload) -> Path | None:
    """Uploaded ground-truth recording as 16 kHz mono WAV (the rate every spectrogram uses)."""
    data = upload.getvalue()
    path = RECORDINGS_DIR / f"{hashlib.sha1(data).hexdigest()[:16]}.wav"
    if path.exists():
        return path
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / f"recording{Path(upload.name).suffix.lower()}"
        raw.write_bytes(data)
        try:
            audio = load_audio(raw, GENERATOR_SAMPLE_RATE)
        except RuntimeError as error:  # soundfile cannot decode it
            st.sidebar.error(f"Could not read {upload.name}: {error}")
            return None
    save_audio(path, audio, GENERATOR_SAMPLE_RATE)
    return path


# -----------------------------------------------------------------------------
# Columns of the board
# -----------------------------------------------------------------------------


def load_examples() -> list[dict]:
    """Figure 5 examples with their real recordings, if the data is present."""
    manifest = EXAMPLES_DIR / "manifest.json"
    return json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []


def example_item(entry: dict) -> dict:
    return {
        "id": entry["id"],
        "title": entry["label"].title(),
        "image": (EXAMPLES_DIR / entry["image"]).read_bytes(),
        "source": f"AudioCaps {entry['id']}",
        "gt_audio": EXAMPLES_DIR / entry["gt_audio"],
        "gt_window": GT_WINDOW,  # 5 s of the 10 s clip, centred on the frame, as in the paper figures
        "gt_captions": entry["captions"],
    }


def upload_item(upload, recording: Path | None, caption: str) -> dict:
    image = upload.getvalue()
    stem = Path(upload.name).stem.replace("_", " ").replace("-", " ").strip()
    return {
        "id": f"upload-{hashlib.sha1(image).hexdigest()[:10]}",
        "title": stem.title() or "Upload",
        "image": image,
        "source": f"uploaded image '{upload.name}'",
        "gt_audio": recording,
        "gt_window": None,  # the whole recording
        "gt_captions": [caption] if caption else [],
    }


# -----------------------------------------------------------------------------
# Board cells
# -----------------------------------------------------------------------------


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def board_row(label_html: str, n: int) -> list:
    """One row: label column plus one cell per image (padded so few images stay narrow)."""
    spec = [1.15] + [1] * n + ([MIN_GRID_UNITS - n] if n < MIN_GRID_UNITS else [])
    columns = st.columns(spec, gap="small", vertical_alignment="center")
    columns[0].markdown(label_html, unsafe_allow_html=True)
    return columns[1 : n + 1]


def label(text: str, sub: str = "", colour: str = "inherit", lead: str = "") -> str:
    lead_html = f"<span class='ours'>{lead}</span>" if lead else ""
    sub_html = f"<small>{sub}</small>" if sub else ""
    return f"<div class='casv-label' style='color:{colour}'>{lead_html}{text}{sub_html}</div>"


def note(cell, text: str) -> None:
    cell.markdown(f"<div class='casv-note'>{text}</div>", unsafe_allow_html=True)


def audio_panel(cell, path: Path | None, empty_text: str, window: tuple | None = None) -> None:
    """Spectrogram and player, or a grey panel saying why there is none."""
    if path is None or not path.exists():
        cell.markdown(f"<div class='casv-empty'>{empty_text}</div>", unsafe_allow_html=True)
        return
    window = tuple(window) if window else None
    cell.image(spectrogram_png(str(path), path.stat().st_mtime, window), width="stretch")
    if window:
        cell.audio(str(path), start_time=window[0], end_time=window[1])
    else:
        cell.audio(str(path))


def export_figure(items: list[dict], config_name: str, config: VerifierConfig, settings: dict) -> Path:
    """Copy each column into a run folder in test.py's layout and render it with scripts/figures.py."""
    run_dir = RUNS_DIR / f"figure_{datetime.now():%Y%m%d_%H%M%S}"
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
        f"CAS-V comparison built in app.py on {datetime.now():%Y-%m-%d}. The images were chosen by hand in "
        f"the app, not sampled: this figure is an illustration, not a result. Existing research: "
        f"{AUDIOLDM_MODEL_ID} prompted with the top-1 retrieved AudioCaps train caption, unverified. Ours: "
        f"the same generator and seed ({settings['seed']}; {settings['duration']} s, {settings['steps']} "
        f"steps) prompted with the CAS-V-verified caption ({config_name} verifier configuration); verified "
        f"concepts and sound type are phrases of that caption. Recognition: {CLIP_MODEL_ID} zero-shot over "
        f"Places365 scenes and ImageNet objects. Target sound: the real recording where one exists (AudioCaps "
        f"examples: 5 s window centred on the input frame; uploads: the whole recording), otherwise marked "
        f"'not provided'. Spectrograms as in scripts/figures.py."
    )
    render_page(run_dir, records, run_dir / "casv_comparison.png", provenance)
    return run_dir


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Images")
    items = []
    examples = load_examples()
    if examples:
        names = [entry["label"] for entry in examples]
        picked = st.multiselect("Figure 5 examples (with real recordings)", names, default=names, format_func=str.title)
        items += [example_item(entry) for entry in examples if entry["label"] in picked]
    else:
        st.caption(
            "Figure 5 examples not found in `data/audiocaps/pairs/categories/` "
            "(created by `py -m scripts.select_examples`)."
        )

    uploads = st.file_uploader("Your images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    if uploads:
        with st.expander("Ground-truth recordings (optional)"):
            st.caption("A real recording of each scene, if you have one. Without it the panel says 'not provided'.")
            for upload in uploads:
                recording_file = st.file_uploader(
                    f"Recording for {upload.name}", type=["wav", "flac", "ogg", "mp3"], key=f"gt_{upload.file_id}"
                )
                caption = st.text_input(
                    "Its caption", key=f"caption_{upload.file_id}", placeholder="e.g. Waves crash on a beach"
                ).strip()
                recording = store_recording(recording_file) if recording_file else None
                items.append(upload_item(upload, recording, caption))

    items = list({item["id"]: item for item in items}.values())  # the same image twice is one column
    if len(items) > MAX_COLUMNS:
        st.warning(f"Showing the first {MAX_COLUMNS} of {len(items)} images.")
        items = items[:MAX_COLUMNS]

    st.header("Verifier")
    configs = ["Default (paper)"] + (["Tuned on val (ablation)"] if TUNED_CONFIG.exists() else [])
    config_name = st.radio("Configuration", configs)
    config = (
        VerifierConfig(**json.loads(TUNED_CONFIG.read_text(encoding="utf-8")))
        if config_name.startswith("Tuned")
        else VerifierConfig()
    )

    st.header("Generation")
    settings = {
        "seed": int(st.number_input("Seed", value=42, step=1)),
        "duration": st.slider("Duration (s)", 2.0, 10.0, 5.0, 0.5, help="The paper figures use 5 s."),
        "steps": st.slider("Diffusion steps", 10, 100, 50, 5, help="Fewer steps are faster but lower quality."),
    }
    st.caption(f"Recognition: `{CLIP_MODEL_ID}`  \nGenerator: `{AUDIOLDM_MODEL_ID}`")


# -----------------------------------------------------------------------------
# Main page
# -----------------------------------------------------------------------------

st.title("CAS-V comparison board")
st.caption(
    "One column per image. **Existing research**: AudioLDM prompted with the top-1 retrieved AudioCaps caption, "
    "unverified. **Ours (CAS-V)**: the same generator and seed prompted with the verified caption. "
    "**Target sound**: the real recording, when there is one."
)

if not items:
    st.info("Pick Figure 5 examples or upload images in the sidebar.")
    st.stop()

verifier = load_verifier()
with st.spinner("Recognising images..."):
    for item in items:
        item["result"] = verifier.decide(measure_image(item["image"], verifier), config)
        item["concepts"] = verified_concepts(item["result"]["casv_prompt"])
        item["wav"] = {arm: clip_path(item["result"][f"{arm}_prompt"], settings) for arm in ("baseline", "casv")}

clips = {item["wav"][arm]: item["result"][f"{arm}_prompt"] for item in items for arm in ("baseline", "casv")}
todo = {path: prompt for path, prompt in clips.items() if not path.exists()}

button_col, info_col = st.columns([1, 3], vertical_alignment="center")
button_label = f"Generate {len(todo)} clip{'s' * (len(todo) != 1)}" if todo else "All audio generated"
if button_col.button(button_label, type="primary", disabled=not todo, width="stretch"):
    generator = load_generator()
    progress = st.progress(0.0)
    for done, (path, prompt) in enumerate(todo.items()):
        progress.progress(done / len(todo), text=f"Clip {done + 1} of {len(todo)}: “{prompt}”")
        generate_clip(generator, prompt, settings, path)
    st.rerun()
shared = 2 * len(items) - len(clips)
info_col.caption(
    f"{len(clips)} clips for {len(items)} images"
    + (f" ({shared} prompt{'s' * (shared != 1)} shared, e.g. CAS-V kept the baseline caption)" if shared else "")
    + ". About 1-2 min per clip on CPU, seconds on a GPU. Clips are cached by prompt and settings, "
    "so changing the seed, duration or steps needs new clips."
)

n = len(items)

for cell, item in zip(board_row(label("Input Image"), n), items, strict=True):
    cell.markdown(
        f"<div class='casv-title' title='{esc(item['title'])}'>{esc(item['title'])}</div>", unsafe_allow_html=True
    )
    cell.image(thumbnail(item["image"]), width="stretch")

for cell, item in zip(board_row(label("Existing Research", "(Without CAS-V)", RED), n), items, strict=True):
    audio_panel(cell, item["wav"]["baseline"], "not generated yet")
    note(cell, f"Prompt: “{esc(item['result']['baseline_prompt'])}”")

with st.container(key="ours"):
    concepts_label = label("Semantic Text", "(Verified Concepts)", BLUE, lead="Ours (CAS-V)")
    for cell, item in zip(board_row(concepts_label, n), items, strict=True):
        result = item["result"]
        prompt = esc(f"CAS-V prompt: {result['casv_prompt']}")
        concepts = esc(", ".join(item["concepts"]))
        cell.markdown(f"<div class='casv-box casv-concepts' title='{prompt}'>{concepts}</div>", unsafe_allow_html=True)
        dot = f"<span class='casv-dot' style='color:{STATUS_COLOURS[result['status']]}'>●</span>"
        note(cell, f"{dot} {result['status']} · {result['n_rejected']} of {len(result['verdicts'])} rejected")

    for cell, item in zip(board_row(label("Spectrogram", "(Generated Audio)", BLUE), n), items, strict=True):
        audio_panel(cell, item["wav"]["casv"], "not generated yet")

    for cell, item in zip(board_row(label("Caption", "(Sound Type)", BLUE), n), items, strict=True):
        cell.markdown(
            f"<div class='casv-box casv-type'>{esc(sound_type(item['concepts']))}</div>", unsafe_allow_html=True
        )

for cell, item in zip(board_row(label("Target Sound", "(Ground Truth)"), n), items, strict=True):
    empty = "not provided" + ("<br>(add a recording in the sidebar)" if item["id"].startswith("upload-") else "")
    audio_panel(cell, item["gt_audio"], empty, item["gt_window"])
    if item["gt_captions"]:
        note(cell, esc(item["gt_captions"][0]))

# -----------------------------------------------------------------------------
# Figure export and details
# -----------------------------------------------------------------------------

st.divider()
st.subheader("Figure")
ready = not todo
if st.button("Build figure from this board", disabled=not ready, help=None if ready else "Generate the audio first."):
    with st.spinner("Rendering..."):
        st.session_state["figure_dir"] = str(export_figure(items, config_name, config, settings))

if "figure_dir" in st.session_state:
    figure_dir = Path(st.session_state["figure_dir"])
    png, caption_txt = figure_dir / "casv_comparison.png", figure_dir / "casv_comparison_caption.txt"
    st.image(str(png), width="stretch")
    download_png, download_txt, _ = st.columns([1, 1, 2])
    download_png.download_button("Download PNG", png.read_bytes(), png.name, "image/png", width="stretch")
    download_txt.download_button(
        "Download caption", caption_txt.read_bytes(), caption_txt.name, "text/plain", width="stretch"
    )
    st.caption(
        f"Saved with every clip and record in `{figure_dir.relative_to(ROOT)}`. Hand-picked images: "
        "label it as an illustration, not a result (the caption file says so)."
    )

st.subheader("Recognition and verification details")
for tab, item in zip(st.tabs([item["title"] for item in items]), items, strict=True):
    result = item["result"]
    with tab:
        image_col, info = st.columns([1, 2])
        image_col.image(item["image"], width="stretch")
        image_col.caption(item["source"])
        env = result["env_probs"]
        info.progress(env["indoor"], text=f"Indoor {env['indoor']:.0%} · outdoor {env['outdoor']:.0%}")
        scenes, objects = info.columns(2)
        scenes.dataframe([{"Places365 scene": s, "p": round(p, 3)} for s, p in result["top_scenes"]], hide_index=True)
        objects.dataframe([{"ImageNet object": o, "p": round(p, 3)} for o, p in result["top_objects"]], hide_index=True)
        info.markdown("**Existing research (without CAS-V)**: top-1 retrieved caption")
        info.info(result["baseline_prompt"])
        info.markdown(f"**Ours (CAS-V)**: verified caption · status **{result['status']}**")
        info.success(result["casv_prompt"])
        st.dataframe(
            [
                {
                    "caption": v["caption"],
                    "visual": round(v["visual"], 2),
                    "scene": round(v["scene_support"], 2),
                    "object": round(v["object_support"], 2),
                    "score": round(v["score"], 3),
                    "decision": v["decision"],
                    "conflicts": "; ".join(v["conflicts"]),
                }
                for v in result["verdicts"]
            ],
            hide_index=True,
            width="stretch",
        )
