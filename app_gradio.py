# =============================================================================
# CAS-V comparison board in Gradio, for a free Hugging Face Space on ZeroGPU
# (ZeroGPU Spaces run Gradio apps only). Same pipeline and board as app.py.
#
#   Recognition and verification  casv/ library, on CPU
#   Generation                    AudioLDM inside @spaces.GPU: a shared GPU, billed
#                                 to the visitor's daily ZeroGPU quota
#   Board                         scripts/figures.py, laid out like docs/target_figure.png
#   Listen                        one card per image: existing research, ours, target sound
#
# Locally (no ZeroGPU) the decorator does nothing and everything runs on CPU;
# generated clips are shared with app.py through app_data/runs/clips/, and the
# Figure 5 examples can be loaded with one click. On a Space everything lives in
# a temporary folder (deleted when it restarts) and the examples, whose
# recordings are YouTube audio, are not published.
#
# Usage:
#   python app_gradio.py          # local: http://127.0.0.1:7860
#   python app_gradio.py --share  # plus a public https://....gradio.live link from this PC
#   py -m scripts.deploy_space <user>/<space>   # put it on a Hugging Face Space
# =============================================================================

import spaces  # noqa: I001  (first: on ZeroGPU it must be imported before torch initialises CUDA)

import argparse
import hashlib
import html
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
ON_ZEROGPU = SpacesConfig.zero_gpu
DEVICE = "cuda" if ON_ZEROGPU or torch.cuda.is_available() else "cpu"
if ON_ZEROGPU:
    WORK_DIR = Path(tempfile.gettempdir()) / "casv_board"
    CLIPS_DIR = WORK_DIR / "clips"
else:  # git-ignored; clips shared with app.py (same model, prompt and settings give the same audio)
    WORK_DIR = ROOT / "app_data" / "runs" / "gradio"
    CLIPS_DIR = ROOT / "app_data" / "runs" / "clips"
RECORDINGS_DIR = WORK_DIR / "recordings"
LISTEN_DIR = WORK_DIR / "listen"  # loudness-matched copies for the players
EXAMPLES_DIR = ROOT / "data" / "audiocaps" / "pairs" / "categories"  # local only, never in the Space bundle
GRAPHS_DIR = ROOT / "journal_results" / "graphs"  # the paper's evaluation graphs, under "More info"
TUNED_CONFIG = ROOT / "results" / "tuning_test400" / "casv_tuned_config.json"
REPO_URL = "https://github.com/Patiencewantae123/CAS-CLAP-2"
MAX_IMAGES = 7  # as in the target figure
STATUS_COLOURS = {"ACCEPT": "#1baf7a", "REPAIR": "#eda100", "REJECT": "#e34948", "UNCERTAIN": "#6250d6"}

CONFIGS = {"Default (paper)": VerifierConfig()}
if TUNED_CONFIG.exists():
    CONFIGS["Tuned on val (ablation)"] = VerifierConfig(**json.loads(TUNED_CONFIG.read_text(encoding="utf-8")))
EXAMPLES = board.load_examples(EXAMPLES_DIR)
EXAMPLE_BY_HASH = {hashlib.sha1((EXAMPLES_DIR / e["image"]).read_bytes()).hexdigest(): e for e in EXAMPLES}

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


def cpu_seconds(n_clips: int, settings: dict) -> float:
    """Rough CPU time: measured on the development PC, 17 s per 5 s / 50-step clip and 2.5 s per 2 s / 10-step clip."""
    return n_clips * (1.5 + 0.065 * settings["steps"] * settings["duration"])


def about(seconds: float) -> str:
    return f"about {round(seconds)} s" if seconds < 90 else f"about {round(seconds / 60)} min"


def generate_missing(prompts: list[str], settings: dict, progress) -> None:
    """Generate the clips not cached yet: one GPU call on ZeroGPU, one clip at a time (with progress) on CPU."""
    todo = [p for p in dict.fromkeys(prompts) if not board.clip_path(CLIPS_DIR, p, settings).exists()]
    batches = [todo] if ON_ZEROGPU else [[prompt] for prompt in todo]
    done = 0
    for batch in filter(None, batches):
        where = "on the GPU" if ON_ZEROGPU else f"on {DEVICE.upper()}, {about(cpu_seconds(1, settings))} each"
        progress(0.1 + 0.8 * done / len(todo), desc=f"Clip {done + 1} of {len(todo)} ({where})")
        clips = generate_clips(batch, settings["seed"], settings["duration"], settings["steps"])
        for prompt, (audio, seconds) in zip(batch, clips, strict=True):
            board.save_clip(board.clip_path(CLIPS_DIR, prompt, settings), audio, prompt, settings, seconds)
        done += len(batch)


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


def recognise(image_files: list[str], recording_files: list[str], config: VerifierConfig, settings: dict) -> list:
    """Board columns with recognition, verification and clip paths; Figure 5 frames keep their provenance."""
    recordings = match_recordings(image_files, recording_files)
    items = {}
    for path in map(Path, image_files):
        data = path.read_bytes()
        entry = EXAMPLE_BY_HASH.get(hashlib.sha1(data).hexdigest())
        if entry:
            item = board.example_item(entry, EXAMPLES_DIR)
        else:
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
    return list(items.values())


def summary_table(items: list[dict]) -> list[list[str]]:
    return [
        [
            item["title"],
            item["result"]["status"],
            f"{item['result']['n_rejected']} of {len(item['result']['verdicts'])}",
            f"{item['result']['top_scenes'][0][0]} / {item['result']['top_objects'][0][0]}",
            item["result"]["baseline_prompt"],
            item["result"]["casv_prompt"],
            ", ".join(item["concepts"]),
            sound_type(item["concepts"]),
            "recording" if item["gt_audio"] else "not provided",
        ]
        for item in items
    ]


def card(item: dict, run_dir: Path) -> dict:
    """What the Listen tab shows for one image (files from the run folder)."""
    result, item_dir = item["result"], run_dir / item["id"]
    return {
        "title": item["title"],
        "image": str(item_dir / "input.png"),
        "status": result["status"],
        "rejected": f"{result['n_rejected']} of {len(result['verdicts'])} candidates rejected",
        "recognised": f"{result['top_scenes'][0][0]} / {result['top_objects'][0][0]}",
        "baseline_prompt": result["baseline_prompt"],
        "casv_prompt": result["casv_prompt"],
        "concepts": item["concepts"],
        "sound_type": sound_type(item["concepts"]),
        # players get loudness-matched copies (board.LISTEN_NOTE); the run folder keeps the originals
        "baseline": str(board.listening_copy(item_dir / "baseline.wav", LISTEN_DIR)),
        "casv": str(board.listening_copy(item_dir / "casv.wav", LISTEN_DIR)),
        "gt": str(board.listening_copy(item_dir / "gt.wav", LISTEN_DIR, item["gt_window"]))
        if (item_dir / "gt.wav").exists()
        else None,
        "gt_caption": item["gt_captions"][0] if item["gt_captions"] else "",
        "candidates": [
            [v["caption"], round(v["score"], 3), v["decision"], "; ".join(v["conflicts"])]
            for v in sorted(result["verdicts"], key=lambda v: v["score"], reverse=True)
        ],
    }


def run(image_files, recording_files, config_name, seed, duration, steps, progress=gr.Progress()):  # noqa: B008 (Gradio's progress API)
    """Generator: yields the recognition table first, then the board, the cards and the download."""
    if not image_files:
        raise gr.Error("Upload at least one image" + (", or load the Figure 5 examples." if EXAMPLES else "."))
    if len(image_files) > MAX_IMAGES:
        gr.Warning(f"Using the first {MAX_IMAGES} of {len(image_files)} images.")
        image_files = image_files[:MAX_IMAGES]
    settings = {"seed": int(seed), "duration": float(duration), "steps": int(steps)}
    config = CONFIGS[config_name]

    progress(0.02, desc="Recognising images")
    items = recognise(image_files, recording_files or [], config, settings)
    if not items:
        raise gr.Error("None of the uploads could be read as an image.")
    prompts = [item["result"][f"{arm}_prompt"] for item in items for arm in ("baseline", "casv")]
    todo = [p for p in dict.fromkeys(prompts) if not board.clip_path(CLIPS_DIR, p, settings).exists()]
    if not todo:
        plan = "All clips were generated before; building the board."
    elif ON_ZEROGPU:
        reserve = gpu_seconds(todo, settings["seed"], settings["duration"], settings["steps"])
        plan = f"Generating {len(todo)} clips on a shared GPU (about {reserve} s of your daily ZeroGPU quota)."
    else:
        plan = f"Generating {len(todo)} clips on {DEVICE.upper()}: {about(cpu_seconds(len(todo), settings))}."
    n = len(items)
    yield f"**Recognised {n} image{'s' * (n != 1)}.** {plan}", summary_table(items), None, None, []

    generate_missing(prompts, settings, progress)

    progress(0.92, desc="Rendering the board")
    run_dir = WORK_DIR / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    figure = board.export_figure(items, run_dir, config_name, config, settings, made_in="the CAS-V Gradio app")
    archive = shutil.make_archive(str(run_dir), "zip", run_dir)
    done = (
        f"**Done: {n} column{'s' * (n != 1)}.** Listen to each image in the **Listen** tab. Hand-picked images make "
        "the board an illustration, not a result; the caption file in the download says so."
    )
    yield done, summary_table(items), str(figure), archive, [card(item, run_dir) for item in items]


def load_examples():
    """The Figure 5 frames, named by category (recognised by content, so their real recordings come along)."""
    folder = WORK_DIR / "examples"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for entry in EXAMPLES:
        path = folder / f"{entry['label'].replace(' / ', ' & ')}.jpg"
        shutil.copyfile(EXAMPLES_DIR / entry["image"], path)
        paths.append(str(path))
    return paths, None


# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def card_header(c: dict) -> str:
    dot = f"<span style='color:{STATUS_COLOURS.get(c['status'], '#888')}'>●</span>"
    return (
        f"<div class='casv-card-title'><h3>{esc(c['title'])}</h3>"
        f"<span class='casv-meta'>{dot} {c['status']} · {esc(c['rejected'])} · recognised: "
        f"{esc(c['recognised'])}</span></div>"
    )


# Each card column starts with a fixed-height header (casv-head) so the three players line up.


def baseline_html(c: dict) -> str:
    return (
        "<div class='casv-head'><div class='casv-label casv-red'>Existing research <span>(without CAS-V)</span></div>"
        f"<div class='casv-prompt'>“{esc(c['baseline_prompt'])}”</div></div>"
    )


def ours_html(c: dict) -> str:
    chips = "".join(f"<span class='casv-chip'>{esc(concept)}</span>" for concept in c["concepts"])
    return (
        "<div class='casv-head'><div class='casv-label casv-blue'>Ours <span>(CAS-V)</span></div>"
        f"<div class='casv-prompt' title='{esc(c['casv_prompt'])}'>{chips}</div>"
        f"<div class='casv-type-row'>Sound type: <span class='casv-type'>{esc(c['sound_type'])}</span></div></div>"
    )


def target_html(c: dict) -> str:
    caption = f"<div class='casv-prompt'>{esc(c['gt_caption'])}</div>" if c["gt_caption"] else ""
    empty = "" if c["gt"] else "<div class='casv-empty'>not provided</div>"
    return (
        "<div class='casv-head'><div class='casv-label'>Target sound <span>(ground truth)</span></div>"
        f"{caption}</div>{empty}"
    )


HEADER = """<div class="casv-hero">
  <h1>CAS-V comparison board</h1>
  <p>Conflict-aware semantic verification for image-to-audio generation. For each image, the same generator and
  seed are prompted twice: with the top-1 retrieved caption, unverified (<b class="casv-red">existing research</b>),
  and with a caption checked against the image (<b class="casv-blue">ours, CAS-V</b>).</p>
</div>"""

PRIVACY = (
    "Audio is generated on a shared GPU and uses your daily ZeroGPU quota. Uploads and results stay in this Space's "
    "temporary storage and are deleted when it restarts."
    if ON_ZEROGPU
    else "Running on this computer; uploads and results are saved under app_data/runs/."
)
HOW_IT_WORKS = f"""**Pipeline.** CLIP (`{CLIP_MODEL_ID}`) recognises the scene (Places365) and objects (ImageNet).
Twenty candidate sound captions are retrieved from the AudioCaps train split; captions that conflict with the image
(a sound tied to the wrong indoor/outdoor setting, or a sound source that is not visible) are rejected and the rest
are scored. Both prompts go to AudioLDM (`{AUDIOLDM_MODEL_ID}`) with the same seed. The board is laid out like the
paper's figures; the download holds it with a caption file giving the provenance of every panel, the clips and the
per-image records.

**Real recordings.** Upload a recording with the same file name as its image (`beach.jpg` + `beach.wav`) to show it
as the target sound.

**What this shows.** What the method does on images you choose: hand-picked images make the board an illustration,
not a result. On 400 AudioCaps test clips, CAS-V showed no measurable CLAP gain over the baseline (+0.003, 95% CI
[-0.012, +0.019]); a blind listening test is in preparation. Protocol, code and results: [{REPO_URL}]({REPO_URL})

**Privacy.** {PRIVACY}"""

HEADERS = [
    "Image",
    "Status",
    "Rejected",
    "Recognised scene / object",
    "Existing research prompt (unverified)",
    "CAS-V prompt (verified)",
    "Verified concepts",
    "Sound type",
    "Target sound",
]

CSS = """
.casv-hero h1 { margin: 0 0 .3rem; font-size: 1.9rem; }
.casv-hero p { margin: 0; opacity: .85; max-width: 80ch; }
.casv-red { color: #d9483b; } .casv-blue { color: #4a72d6; }
.casv-card { padding: .8rem 1rem !important; margin-bottom: .9rem; }
.casv-card-title { display: flex; gap: .8rem; align-items: baseline; flex-wrap: wrap; margin-bottom: .4rem; }
.casv-card-title h3 { margin: 0; font-size: 1.15rem; }
.casv-meta { font-size: .85rem; opacity: .8; }
.casv-label { font-weight: 700; font-size: .95rem; margin-bottom: .3rem; }
.casv-label span { font-weight: 600; font-size: .8rem; opacity: .8; }
.casv-head { min-height: 8.4rem; }
.casv-prompt { font-size: .85rem; opacity: .9; margin-bottom: .45rem; line-height: 1.35; }
.casv-chip { display: inline-block; background: #edf5ea; border: 1px solid #8fbf7f; color: #1f3a93;
             border-radius: 999px; padding: 1px 9px; margin: 0 4px 4px 0; font-size: .8rem; }
.casv-type-row { font-size: .8rem; margin-bottom: .45rem; opacity: .95; }
.casv-type { display: inline-block; background: #eaf0fb; border: 1px solid #8aa6d6; color: #1f3a93;
             border-radius: 6px; padding: 1px 8px; font-weight: 600; }
.casv-caption { font-size: .9rem; opacity: .85; max-width: 95ch; margin-bottom: 1.4rem; }
.casv-empty { border-radius: 6px; background: rgba(128, 128, 128, .13); color: rgba(128, 128, 128, .95);
              padding: 1.4rem .5rem; text-align: center; font-size: .85rem; }
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="green",
    neutral_hue="slate",
    font=["ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
    font_mono=["ui-monospace", "Consolas", "monospace"],
)

with gr.Blocks(title="CAS-V comparison board", fill_width=True) as demo:
    gr.HTML(HEADER)
    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            images = gr.File(label=f"Images (up to {MAX_IMAGES})", file_count="multiple", file_types=["image"])
            recordings = gr.File(
                label="Real recordings (optional, same file name as the image)",
                file_count="multiple",
                file_types=["audio"],
            )
            examples_button = gr.Button(
                f"Load the {len(EXAMPLES)} Figure 5 examples (with real recordings)",
                size="md",
                visible=bool(EXAMPLES),
            )
            with gr.Accordion("Settings", open=False):
                config_name = gr.Radio(list(CONFIGS), value="Default (paper)", label="Verifier configuration")
                seed = gr.Number(42, precision=0, label="Seed")
                duration = gr.Slider(2, 10, value=5, step=0.5, label="Duration (s)", info="The paper figures use 5 s.")
                steps = gr.Slider(10, 100, value=50, step=5, label="Diffusion steps", info="Fewer: faster, rougher.")
            button = gr.Button("Build the board", variant="primary", size="lg")
            status = gr.Markdown("Add images, then build the board.")
        with gr.Column(scale=3), gr.Tabs():
            with gr.Tab("Board"):
                figure = gr.Image(label="Comparison board", type="filepath", interactive=False, buttons=["fullscreen"])
                download = gr.File(label="Download: board, caption, clips and records (zip)", height=110)
            with gr.Tab("Listen"):
                cards = gr.State([])

                @gr.render(inputs=cards)
                def show_cards(cards_value):
                    if not cards_value:
                        gr.Markdown("Build the board to listen to every image here, side by side.")
                        return
                    gr.Markdown(board.LISTEN_NOTE, elem_classes="casv-caption")
                    for c in cards_value:
                        with gr.Group(elem_classes="casv-card"):
                            gr.HTML(card_header(c))
                            with gr.Row(equal_height=False):
                                gr.Image(
                                    c["image"],
                                    show_label=False,
                                    container=False,
                                    height=170,
                                    interactive=False,
                                    scale=1,
                                    min_width=180,
                                    buttons=[],
                                )
                                for html_block, audio in (
                                    (baseline_html(c), c["baseline"]),
                                    (ours_html(c), c["casv"]),
                                    (target_html(c), c["gt"]),
                                ):
                                    with gr.Column(scale=2, min_width=220):
                                        gr.HTML(html_block)
                                        if audio:
                                            gr.Audio(audio, show_label=False, interactive=False, buttons=["download"])
                            with gr.Accordion("All candidates and their verification scores", open=False):
                                gr.Dataframe(
                                    value=c["candidates"],
                                    headers=["Candidate caption", "Score", "Decision", "Conflicts"],
                                    column_widths=["50%", "10%", "12%", "28%"],
                                    wrap=True,
                                    interactive=False,
                                    show_label=False,
                                )

            with gr.Tab("Details"):
                table = gr.Dataframe(
                    headers=HEADERS, wrap=True, interactive=False, label="Recognition and verification"
                )
    with gr.Accordion("More info: how it works, what it shows, and the paper's results", open=False):
        gr.Markdown(HOW_IT_WORKS)
        graphs = [(name, title, caption) for name, title, caption in board.PAPER_GRAPHS if (GRAPHS_DIR / name).exists()]
        if graphs:
            gr.Markdown(f"### Paper results\n{board.PAPER_RESULTS_INTRO}")
            for name, title, caption in graphs:
                gr.Markdown(f"#### {title}")
                gr.Image(
                    str(GRAPHS_DIR / name),
                    show_label=False,
                    interactive=False,
                    container=False,
                    height=380,
                    buttons=["fullscreen"],
                )
                gr.Markdown(caption, elem_classes="casv-caption")

    examples_button.click(load_examples, outputs=[images, recordings])
    button.click(
        run,
        inputs=[images, recordings, config_name, seed, duration, steps],
        outputs=[status, table, figure, download, cards],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAS-V comparison board (Gradio).")
    parser.add_argument(
        "--share",
        action="store_true",
        help="also print a public https://....gradio.live link, served from this PC while it runs (about a week)",
    )
    demo.queue().launch(
        allowed_paths=[str(WORK_DIR), str(CLIPS_DIR), str(EXAMPLES_DIR), str(GRAPHS_DIR)],
        share=parser.parse_args().share,
        theme=THEME,
        css=CSS,
    )
