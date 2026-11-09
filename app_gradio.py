# =============================================================================
# CAS-V comparison board in Gradio, for a free Hugging Face Space on ZeroGPU
# (ZeroGPU Spaces run Gradio apps only). Same pipeline and board as app.py.
#
#   Recognition and verification  casv/ library, on CPU
#   Generation                    AudioLDM inside @spaces.GPU: a shared GPU, billed
#                                 to the visitor's daily ZeroGPU quota
#   Board                         scripts/figures.py, laid out like docs/target_figure.png
#
# Locally (no ZeroGPU) the decorator does nothing and everything runs on CPU.
# Uploads, clips and boards go to a temporary folder; on a Space they are
# deleted when it restarts.
#
# Usage:
#   python app_gradio.py          # local: http://127.0.0.1:7860
#   python app_gradio.py --share  # plus a public https://....gradio.live link from this PC
#   py -m scripts.build_space     # bundle for the Space (README: "Hosting")
# =============================================================================

import spaces  # noqa: I001  (first: on ZeroGPU it must be imported before torch initialises CUDA)

import argparse
import json
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from spaces.config import Config as SpacesConfig

import board
from casv.audio import AUDIOLDM_MODEL_ID, AudioGenerator
from casv.log import setup_logging
from casv.text import sound_type, verified_concepts
from casv.verifier import CLIP_MODEL_ID, CASVerifier, CLIPRecognizer, VerifierConfig

ROOT = Path(__file__).resolve().parent
WORK_DIR = Path(tempfile.gettempdir()) / "casv_board"
CLIPS_DIR = WORK_DIR / "clips"
RECORDINGS_DIR = WORK_DIR / "recordings"
TUNED_CONFIG = ROOT / "results" / "tuning_test400" / "casv_tuned_config.json"
REPO_URL = "https://github.com/Patiencewantae123/CAS-CLAP-2"
MAX_IMAGES = 7  # as in the target figure
ON_ZEROGPU = SpacesConfig.zero_gpu
DEVICE = "cuda" if ON_ZEROGPU or torch.cuda.is_available() else "cpu"

CONFIGS = {"Default (paper)": VerifierConfig()}
if TUNED_CONFIG.exists():
    CONFIGS["Tuned on val (ablation)"] = VerifierConfig(**json.loads(TUNED_CONFIG.read_text(encoding="utf-8")))

setup_logging()
VERIFIER = CASVerifier(CLIPRecognizer())  # CPU
GENERATOR = AudioGenerator(device=DEVICE)  # ZeroGPU wants models on cuda at start-up


# -----------------------------------------------------------------------------
# Generation (GPU on ZeroGPU)
# -----------------------------------------------------------------------------


def gpu_seconds(prompts, seed, duration, steps):
    """GPU time to reserve for a request: a generous per-clip estimate, so short runs stay cheap."""
    return int(10 + len(prompts) * (2 + 0.02 * steps * duration))


@spaces.GPU(duration=gpu_seconds)
def generate_clips(prompts, seed, duration, steps):
    """AudioLDM clips with the same seed for every prompt; returns (audio, seconds) pairs."""
    clips = []
    for prompt in prompts:
        start = time.time()
        audio = GENERATOR.generate(prompt, seed=seed, duration=duration, steps=steps)
        clips.append((audio, time.time() - start))
    return clips


# -----------------------------------------------------------------------------
# One run: recognise, verify, generate, render
# -----------------------------------------------------------------------------


def match_recordings(image_files: list[str], recording_files: list[str]) -> dict:
    """Recordings named like an image (beach.jpg + beach.wav) become its ground truth."""
    stems = {Path(path).stem.lower() for path in image_files}
    matched = {}
    for path in map(Path, recording_files):
        if path.stem.lower() not in stems:
            gr.Warning(f"{path.name}: no image has the same name, so it is not used.")
            continue
        try:
            matched[path.stem.lower()] = board.convert_recording(path.read_bytes(), path.name, RECORDINGS_DIR)
        except RuntimeError:
            gr.Warning(f"{path.name} could not be read as audio.")
    return matched


def summary_table(items: list[dict]) -> list[list[str]]:
    return [
        [
            item["title"],
            item["result"]["status"],
            f"{item['result']['n_rejected']} of {len(item['result']['verdicts'])}",
            item["result"]["baseline_prompt"],
            item["result"]["casv_prompt"],
            ", ".join(item["concepts"]),
            sound_type(item["concepts"]),
            "recording" if item["gt_audio"] else "not provided",
        ]
        for item in items
    ]


def run(image_files, recording_files, config_name, seed, duration, steps, progress=gr.Progress()):  # noqa: B008 (Gradio's progress API)
    """Generator: yields the recognition table first, then the board, players and download."""
    if not image_files:
        raise gr.Error("Upload at least one image.")
    if len(image_files) > MAX_IMAGES:
        gr.Warning(f"Using the first {MAX_IMAGES} of {len(image_files)} images.")
        image_files = image_files[:MAX_IMAGES]
    settings = {"seed": int(seed), "duration": float(duration), "steps": int(steps)}
    config = CONFIGS[config_name]
    recordings = match_recordings(image_files, recording_files or [])

    progress(0.05, desc="Recognising images")
    items = {}
    for path in map(Path, image_files):
        data = path.read_bytes()
        item = board.upload_item(path.name, data, recordings.get(path.stem.lower()))
        if item["id"] in items:
            continue  # the same image twice is one column
        try:
            image = board.open_image(data)
        except OSError:
            gr.Warning(f"{path.name} is not a readable image.")
            continue
        result = VERIFIER.decide(VERIFIER.measure(image), config)
        item["result"] = result
        item["concepts"] = verified_concepts(result["casv_prompt"])
        item["wav"] = {
            arm: board.clip_path(CLIPS_DIR, result[f"{arm}_prompt"], settings) for arm in ("baseline", "casv")
        }
        items[item["id"]] = item
    items = list(items.values())
    if not items:
        raise gr.Error("None of the uploads could be read as an image.")

    prompts = [item["result"][f"{arm}_prompt"] for item in items for arm in ("baseline", "casv")]
    todo = [p for p in dict.fromkeys(prompts) if not board.clip_path(CLIPS_DIR, p, settings).exists()]
    reserve = gpu_seconds(todo, settings["seed"], settings["duration"], settings["steps"])
    where = (
        f"on a shared GPU (reserving about {reserve} s of your daily ZeroGPU quota)"
        if ON_ZEROGPU
        else f"on {DEVICE.upper()} (about 1-2 min per clip on CPU)"
    )
    status = f"Recognised {len(items)} image{'s' * (len(items) != 1)}. " + (
        f"Generating {len(todo)} clip{'s' * (len(todo) != 1)} {where}..." if todo else "Reusing generated clips..."
    )
    yield status, summary_table(items), None, gr.update(choices=[], value=None), None, []

    if todo:
        progress(0.2, desc=f"Generating {len(todo)} clips")
        clips = generate_clips(todo, settings["seed"], settings["duration"], settings["steps"])
        for prompt, (audio, seconds) in zip(todo, clips, strict=True):
            board.save_clip(board.clip_path(CLIPS_DIR, prompt, settings), audio, prompt, settings, seconds)

    progress(0.9, desc="Rendering the board")
    run_dir = WORK_DIR / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    figure = board.export_figure(items, run_dir, config_name, config, settings, made_in="the CAS-V Hugging Face Space")
    archive = shutil.make_archive(str(run_dir), "zip", run_dir)
    players = [
        {
            "baseline": str(item["wav"]["baseline"]),
            "casv": str(item["wav"]["casv"]),
            "gt": str(item["gt_audio"]) if item["gt_audio"] else None,
        }
        for item in items
    ]
    choices = [(f"{n}. {item['title']}", n - 1) for n, item in enumerate(items, 1)]
    done = (
        f"Done: {len(items)} column{'s' * (len(items) != 1)}. Hand-picked images make the board an illustration, "
        "not a result; the caption file in the download says so."
    )
    yield done, summary_table(items), str(figure), gr.update(choices=choices, value=0), archive, players


def listen(index, players):
    """Players for one board column."""
    if index is None or not players:
        return None, None, None
    chosen = players[int(index)]
    return chosen["baseline"], chosen["casv"], chosen["gt"]


# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------

INTRO = f"""# CAS-V comparison board
**Conflict-aware semantic verification for image-to-audio generation.** For each image, CLIP recognises the scene
and objects, 20 candidate sound captions are retrieved from AudioCaps, and captions that conflict with the image
(wrong indoor/outdoor setting, a sound source that is not visible) are rejected.
**Existing research** generates audio from the top-1 retrieved caption, unverified; **Ours (CAS-V)** uses the
verified caption, with the same generator (AudioLDM) and seed. The board is laid out like the paper's figures.

Upload up to {MAX_IMAGES} images. To show a real recording as the target sound, upload it with the same file name
as its image (`beach.jpg` + `beach.wav`)."""

NOTES = f"""Audio is generated on a shared GPU and uses your daily ZeroGPU quota (signed-in Hugging Face users get
more). Uploads and results stay in this Space's temporary storage and are deleted when it restarts.
On 400 AudioCaps test clips, CAS-V showed no measurable CLAP gain over the baseline (+0.003, 95% CI
[-0.012, +0.019]); a listening test is in preparation. This demo shows what the method does, not that it works
better. Recognition: `{CLIP_MODEL_ID}` · generator: `{AUDIOLDM_MODEL_ID}` · code and results: [{REPO_URL}]({REPO_URL})"""

HEADERS = [
    "Image",
    "Status",
    "Rejected",
    "Existing research prompt (unverified)",
    "CAS-V prompt (verified)",
    "Verified concepts",
    "Sound type",
    "Target sound",
]

with gr.Blocks(title="CAS-V comparison board", fill_width=True) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            images = gr.File(label=f"Images (up to {MAX_IMAGES})", file_count="multiple", file_types=["image"])
            recordings = gr.File(
                label="Real recordings (optional, same file name as the image)",
                file_count="multiple",
                file_types=["audio"],
            )
            with gr.Accordion("Settings", open=False):
                config_name = gr.Radio(list(CONFIGS), value="Default (paper)", label="Verifier configuration")
                seed = gr.Number(42, precision=0, label="Seed")
                duration = gr.Slider(2, 10, value=5, step=0.5, label="Duration (s)", info="The paper figures use 5 s.")
                steps = gr.Slider(10, 100, value=50, step=5, label="Diffusion steps")
            button = gr.Button("Build the board", variant="primary")
            status = gr.Markdown()
        with gr.Column(scale=3):
            figure = gr.Image(label="Comparison board", type="filepath", interactive=False)
            table = gr.Dataframe(headers=HEADERS, wrap=True, interactive=False, label="Recognition and verification")
            with gr.Row():
                column = gr.Dropdown(label="Listen to", choices=[], interactive=True)
                download = gr.File(label="Download: board, caption, clips and records (zip)", height=110)
            with gr.Row():
                baseline_audio = gr.Audio(label="Existing research (without CAS-V)", type="filepath", interactive=False)
                casv_audio = gr.Audio(label="Ours (CAS-V)", type="filepath", interactive=False)
                gt_audio = gr.Audio(label="Target sound (ground truth)", type="filepath", interactive=False)
    gr.Markdown(NOTES)

    players = gr.State([])
    button.click(
        run,
        inputs=[images, recordings, config_name, seed, duration, steps],
        outputs=[status, table, figure, column, download, players],
    )
    column.change(listen, inputs=[column, players], outputs=[baseline_audio, casv_audio, gt_audio])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAS-V comparison board (Gradio).")
    parser.add_argument(
        "--share",
        action="store_true",
        help="also print a public https://....gradio.live link, served from this PC while it runs (about a week)",
    )
    demo.queue().launch(allowed_paths=[str(WORK_DIR)], share=parser.parse_args().share)
