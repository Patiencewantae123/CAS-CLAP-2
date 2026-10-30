# =============================================================================
# CAS-V demo (Streamlit): the paper's pipeline, interactively.
#
#   1. Recognition   CLIP zero-shot: Places365 scene, ImageNet object, indoor/outdoor
#   2. Verification  top-20 AudioCaps-train captions scored, conflicts rejected
#   3. Generation    AudioLDM with the same seed for the unverified baseline
#                    prompt and the CAS-V prompt (on demand; slow on CPU)
#
# Every run is saved to app_data/runs/<timestamp>/. The verifier logic is the
# casv/ library used for the reported results; nothing here is simulated.
#
# Usage:
#   streamlit run demo_app.py
# =============================================================================

import json
import time
from dataclasses import asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from casv.audio import AUDIOLDM_MODEL_ID, GENERATOR_SAMPLE_RATE, load_audio, mel_spectrogram_db, save_audio
from casv.text import sound_type, verified_concepts
from casv.verifier import CLIP_MODEL_ID, CASVerifier, CLIPRecognizer, VerifierConfig

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "app_data" / "runs"
EXAMPLES_DIR = ROOT / "data" / "audiocaps" / "pairs" / "categories"
TUNED_CONFIG = ROOT / "results" / "tuning" / "casv_tuned_config.json"
DECISION_COLOURS = {"ACCEPT": "#1baf7a", "REPAIR": "#eda100", "REJECT": "#e34948", "UNCERTAIN": "#6250d6"}

st.set_page_config(page_title="CAS-V demo", page_icon="🔊", layout="wide")


# -----------------------------------------------------------------------------
# Cached models and helpers
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


@st.cache_data(show_spinner=False, max_entries=32)
def measure_image(image_bytes: bytes, _verifier: CASVerifier) -> dict:
    """Config-independent measurement of an image, cached by image content."""
    return _verifier.measure(Image.open(BytesIO(image_bytes)).convert("RGB"))


def load_examples() -> list[dict]:
    """Figure 5 examples with their real recordings, if the data is present."""
    manifest = EXAMPLES_DIR / "manifest.json"
    return json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []


def spectrogram_figure(audio, title: str):
    """Mel spectrogram in the same style as the paper figures."""
    mel = mel_spectrogram_db(audio)
    fig, ax = plt.subplots(figsize=(4, 2.2), dpi=120)
    ax.imshow(mel, origin="lower", aspect="auto", cmap="inferno", vmin=mel.max() - 80, vmax=mel.max())
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def generate_run(
    result: dict, image_bytes: bytes, example: dict | None, config: VerifierConfig, settings: dict
) -> Path:
    """Generate baseline and CAS-V audio with the same seed and save the run."""
    generator = load_generator()
    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    prompts = {"baseline": result["baseline_prompt"], "casv": result["casv_prompt"]}
    timings = {}
    for arm, prompt in prompts.items():
        if arm == "casv" and prompt == prompts["baseline"]:
            # Same prompt and seed give identical audio.
            (run_dir / "casv.wav").write_bytes((run_dir / "baseline.wav").read_bytes())
            continue
        with st.spinner(f"Generating {arm} audio..."):
            start = time.time()
            audio = generator.generate(
                prompt, seed=settings["seed"], duration=settings["duration"], steps=settings["steps"]
            )
            timings[arm] = round(time.time() - start, 1)
        save_audio(run_dir / f"{arm}.wav", audio, GENERATOR_SAMPLE_RATE)
    (run_dir / "input.png").write_bytes(image_bytes)
    record = {
        "timestamp": datetime.now().isoformat(),
        "example": example["id"] if example else None,
        "config": asdict(config),
        "settings": settings,
        "status": result["status"],
        "prompts": prompts,
        "generation_seconds": timings,
    }
    (run_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return run_dir


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Input")
    examples = load_examples()
    sources = ["Upload an image"] + (["Figure 5 example (with real recording)"] if examples else [])
    source = st.radio("Image source", sources, label_visibility="collapsed")

    image_bytes, example = None, None
    if source == "Upload an image":
        upload = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
        if upload is not None:
            image_bytes = upload.getvalue()
    else:
        labels = [e["label"].title() for e in examples]
        example = examples[labels.index(st.selectbox("Example", labels))]
        image_bytes = (EXAMPLES_DIR / example["image"]).read_bytes()

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
        "duration": st.slider("Duration (s)", 2.0, 10.0, 5.0, 0.5),
        "steps": st.slider("Diffusion steps", 10, 100, 50, 5, help="Fewer steps are faster but lower quality."),
    }
    st.caption(f"Recognition: `{CLIP_MODEL_ID}`  \nGenerator: `{AUDIOLDM_MODEL_ID}`")


# -----------------------------------------------------------------------------
# Main page
# -----------------------------------------------------------------------------

st.title("CAS-V: Conflict-Aware Semantic Verification")
st.caption(
    "Image → recognition → verified sound description → generated audio. Same generator and seed for both prompts."
)

if image_bytes is None:
    st.info("Upload an image or pick a Figure 5 example in the sidebar.")
    st.stop()

verifier = load_verifier()
result = verifier.decide(measure_image(image_bytes, verifier), config)

col_image, col_recognition = st.columns([1, 1.3])
with col_image:
    st.image(image_bytes, width="stretch")
with col_recognition:
    status = result["status"]
    st.markdown(
        f"**Status:** <span style='color:{DECISION_COLOURS[status]};font-weight:600'>{status}</span> "
        f"&nbsp;·&nbsp; {result['n_rejected']} of {len(result['verdicts'])} candidates rejected",
        unsafe_allow_html=True,
    )
    env = result["env_probs"]
    st.progress(env["indoor"], text=f"Indoor {env['indoor']:.0%} · outdoor {env['outdoor']:.0%}")
    scenes, objects = st.columns(2)
    scenes.markdown("**Places365 scenes**")
    scenes.dataframe([{"scene": s, "p": round(p, 3)} for s, p in result["top_scenes"]], hide_index=True)
    objects.markdown("**ImageNet objects**")
    objects.dataframe([{"object": o, "p": round(p, 3)} for o, p in result["top_objects"]], hide_index=True)

st.subheader("Sound descriptions")
concepts = verified_concepts(result["casv_prompt"])
baseline_col, casv_col = st.columns(2)
baseline_col.markdown("**Existing research (without CAS-V)**: top-1 retrieved caption")
baseline_col.info(result["baseline_prompt"])
casv_col.markdown("**Ours (CAS-V)**: verified caption")
casv_col.success(result["casv_prompt"])
casv_col.caption(f"Verified concepts: {', '.join(concepts)}  \nSound type: {sound_type(concepts)}")

with st.expander(f"All {len(result['verdicts'])} candidates and verification scores"):
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

st.subheader("Audio")
st.caption("Generated on demand: about 1-2 minutes per clip on CPU, seconds on a GPU.")
if st.button("Generate baseline and CAS-V audio", type="primary"):
    st.session_state["last_run"] = str(generate_run(result, image_bytes, example, config, settings))

if "last_run" in st.session_state:
    run_dir = Path(st.session_state["last_run"])
    panels = [("Existing research (without CAS-V)", run_dir / "baseline.wav"), ("Ours (CAS-V)", run_dir / "casv.wav")]
    if example:
        panels.append(("Target sound (real recording)", EXAMPLES_DIR / example["gt_audio"]))
    for column, (title, path) in zip(st.columns(len(panels)), panels, strict=True):
        if path.exists():
            column.pyplot(spectrogram_figure(load_audio(path, GENERATOR_SAMPLE_RATE), title))
            column.audio(str(path))
    st.caption(f"Saved to `{run_dir.relative_to(ROOT)}`")
