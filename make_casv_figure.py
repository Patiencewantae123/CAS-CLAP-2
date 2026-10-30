# =============================================================================
# Qualitative comparison figure for a CAS-V run.
#
# Rows: input image | baseline spectrogram (without CAS-V) | verified concepts |
#       CAS-V spectrogram | caption | ground-truth spectrogram
#
# Every panel is rendered from files produced by run_casv_experiment.py.
# A missing ground-truth recording is shown as an empty panel, never filled in.
#
# Usage:
#   py make_casv_figure.py --run results/main [--items busy_road,office,...]
# =============================================================================

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from casv.audio import GENERATOR_SAMPLE_RATE, load_audio, mel_spectrogram_db
from casv.concepts import SCENES

ROW_LABELS = [
    ("Input Image", "black"),
    ("Existing Research\n(Without CAS-V)", "#c0392b"),
    ("Semantic Text\n(Verified Concepts)", "#1f3a93"),
    ("Spectrogram\n(Generated Audio)", "#1f3a93"),
    ("Caption\n(Sound Type)", "#1f3a93"),
    ("Target Sound\n(Ground Truth)", "black"),
]
HEIGHT_RATIOS = [1.25, 1.0, 0.7, 1.0, 0.45, 1.0]


def draw_spectrogram(ax, wav_path):
    if not wav_path.exists():
        ax.set_facecolor("#eeeeee")
        ax.text(0.5, 0.5, "not provided", ha="center", va="center", fontsize=8, color="#777777",
                transform=ax.transAxes)
        return
    mel = mel_spectrogram_db(load_audio(wav_path, GENERATOR_SAMPLE_RATE))
    ax.imshow(mel, origin="lower", aspect="auto", cmap="inferno")


def draw_text_box(ax, text, face, edge, fontsize):
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.03, 0.06), 0.94, 0.88, boxstyle="round,pad=0.01,rounding_size=0.03",
                                transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=0.8))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=fontsize, color="#1f3a93",
            transform=ax.transAxes, linespacing=1.3)


def main():
    parser = argparse.ArgumentParser(description="Render the CAS-V qualitative comparison figure.")
    parser.add_argument("--run", required=True, help="Run directory produced by run_casv_experiment.py")
    parser.add_argument("--items", default=None, help="Comma-separated item ids (default: all)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run)
    results = json.loads((run_dir / "results.json").read_text())
    items = results["items"]
    if args.items:
        wanted = args.items.split(",")
        items = [next(r for r in items if r["id"] == item_id) for item_id in wanted]

    n_cols = len(items)
    fig = plt.figure(figsize=(1.0 + 1.75 * n_cols, 9.2), dpi=200)
    grid = fig.add_gridspec(len(ROW_LABELS), n_cols + 1, width_ratios=[0.9] + [1] * n_cols,
                            height_ratios=HEIGHT_RATIOS, wspace=0.06, hspace=0.12)

    for row, (label, color) in enumerate(ROW_LABELS):
        ax = fig.add_subplot(grid[row, 0])
        ax.axis("off")
        ax.text(0.95, 0.5, label, ha="right", va="center", fontsize=8.5, fontweight="bold", color=color)

    for col, record in enumerate(items, start=1):
        item_dir = run_dir / record["id"]
        title = (record["label"] or record["top_scene"] or record["id"]).title()

        ax = fig.add_subplot(grid[0, col])
        image_file = next(item_dir.glob("input.*"))
        ax.imshow(Image.open(image_file).convert("RGB").resize((320, 240)))
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

        for row, wav in ((1, "baseline.wav"), (3, "casv.wav"), (5, "gt.wav")):
            ax = fig.add_subplot(grid[row, col])
            draw_spectrogram(ax, item_dir / wav)
            ax.set_xticks([])
            ax.set_yticks([])
        gt_caption = SCENES[record["label"]]["caption"] if record["label"] else ""
        ax.set_xlabel(textwrap.fill(gt_caption, 22), fontsize=7.5)

        concepts = ",\n".join(textwrap.fill(", ".join(record["verified_concepts"][i:i + 2]), 26)
                              for i in range(0, len(record["verified_concepts"]), 2))
        draw_text_box(fig.add_subplot(grid[2, col]), concepts, "#eef5ea", "#9bbf8a", 7)
        draw_text_box(fig.add_subplot(grid[4, col]), textwrap.fill(record["caption"], 20),
                      "#eaf0fb", "#8aa6d6", 7.5)

    models = results["models"]
    fig.text(0.5, 0.005,
             f"Run '{results['run_name']}' | recognition: {models['clip']} | generator: {models['generator']} "
             f"| seed {results['settings']['seed']} | baseline = same generator prompted with unverified "
             f"top-{results['verifier_config']['max_concepts']} CLIP concepts",
             ha="center", fontsize=6, color="#555555")

    output = Path(args.output) if args.output else run_dir / "casv_comparison.png"
    fig.savefig(output, bbox_inches="tight")
    print(f"[+] Saved figure: {output.resolve()}")


if __name__ == "__main__":
    main()
