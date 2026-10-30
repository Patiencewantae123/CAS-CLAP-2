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
#   py -m scripts.make_casv_figure --run results/audiocaps_test100 [--items <ytid>,<ytid>,...]
# =============================================================================

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from casv.audio import GENERATOR_SAMPLE_RATE, load_audio, mel_spectrogram_db

ROW_LABELS = [
    ("Input Image", "black"),
    ("Existing Research\n(Without CAS-V)", "#c0392b"),
    ("Semantic Text\n(Verified Caption)", "#1f3a93"),
    ("Spectrogram\n(Generated Audio)", "#1f3a93"),
    ("Recognition\n(Scene / Object)", "#1f3a93"),
    ("Target Sound\n(Ground Truth)", "black"),
]
HEIGHT_RATIOS = [1.2, 1.0, 0.95, 1.0, 0.55, 1.0]


def draw_spectrogram(ax, wav_path):
    if not wav_path.exists():
        ax.set_facecolor("#eeeeee")
        ax.text(0.5, 0.5, "not provided", ha="center", va="center", fontsize=8, color="#777777",
                transform=ax.transAxes)
        return
    mel = mel_spectrogram_db(load_audio(wav_path, GENERATOR_SAMPLE_RATE))
    ax.imshow(mel, origin="lower", aspect="auto", cmap="inferno")


def shorten(text, width, max_lines):
    return textwrap.fill(text, width, max_lines=max_lines, placeholder=" ...")


def draw_text_box(ax, text, face, edge, fontsize):
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.03, 0.06), 0.94, 0.88, boxstyle="round,pad=0.01,rounding_size=0.03",
                                transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=0.8))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=fontsize, color="#1f3a93",
            transform=ax.transAxes, linespacing=1.3)


def load_records(run_dir):
    # Read per-item records directly so figures can be made while a run is in
    # progress. Items are kept in processing order (= manifest order).
    records = []
    for path in sorted(run_dir.glob("*/record.json"), key=lambda p: p.stat().st_mtime):
        if (path.parent / "casv.wav").exists():
            records.append(json.loads(path.read_text()))
    return records


def render_page(run_dir, items, output, footer):
    n_cols = len(items)
    fig = plt.figure(figsize=(1.3 + 2.1 * n_cols, 11.0), dpi=200)
    grid = fig.add_gridspec(len(ROW_LABELS), n_cols + 1, width_ratios=[0.9] + [1] * n_cols,
                            height_ratios=HEIGHT_RATIOS, wspace=0.08, hspace=0.42)

    for row, (label, color) in enumerate(ROW_LABELS):
        ax = fig.add_subplot(grid[row, 0])
        ax.axis("off")
        ax.text(0.95, 0.5, label, ha="right", va="center", fontsize=8.5, fontweight="bold", color=color)

    for col, record in enumerate(items, start=1):
        item_dir = run_dir / record["id"]
        title = record["top_scene"][0].title()

        ax = fig.add_subplot(grid[0, col])
        image_file = next(item_dir.glob("input.*"))
        ax.imshow(Image.open(image_file).convert("RGB").resize((320, 240)))
        ax.set_title(textwrap.fill(title, 18), fontsize=9, fontweight="bold")
        ax.axis("off")

        for row, wav in ((1, "baseline.wav"), (3, "casv.wav"), (5, "gt.wav")):
            ax = fig.add_subplot(grid[row, col])
            draw_spectrogram(ax, item_dir / wav)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 1:
                ax.set_xlabel(shorten(f"prompt: {record['prompts']['baseline']}", 34, 2),
                              fontsize=5.5, color="#52514e")
        gt_caption = record["gt_captions"][0] if record["gt_captions"] else ""
        ax.set_xlabel(shorten(gt_caption, 34, 3), fontsize=6)

        draw_text_box(fig.add_subplot(grid[2, col]), shorten(record["prompts"]["casv"], 30, 5),
                      "#eef5ea", "#9bbf8a", 6)
        recognition = (f"{record['top_scene'][0]} / {record['top_object'][0]}\n"
                       f"{record['status']} ({record['n_rejected']} rejected)")
        draw_text_box(fig.add_subplot(grid[4, col]), "\n".join(shorten(line, 30, 2) for line in recognition.split("\n")),
                      "#eaf0fb", "#8aa6d6", 6)

    fig.text(0.5, 0.005, footer, ha="center", fontsize=6, color="#555555")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved figure: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Render the CAS-V qualitative comparison figures.")
    parser.add_argument("--run", required=True, help="Run directory produced by run_casv_experiment.py")
    parser.add_argument("--items", default=None, help="Comma-separated item ids (default: first --count)")
    parser.add_argument("--count", type=int, default=20, help="Number of items when --items is not given")
    parser.add_argument("--per-figure", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42, help="Seed the run used (for the footer)")
    args = parser.parse_args()

    run_dir = Path(args.run)
    records = load_records(run_dir)
    if args.items:
        by_id = {r["id"]: r for r in records}
        records = [by_id[item_id] for item_id in args.items.split(",")]
    else:
        records = records[:args.count]
    if not records:
        raise SystemExit(f"No finished items in {run_dir}")

    from casv.audio import AUDIOLDM_MODEL_ID
    from casv.verifier import CLIP_MODEL_ID
    footer = (f"Run '{run_dir.name}' | recognition: {CLIP_MODEL_ID} (Places365 / ImageNet) | generator: "
              f"{AUDIOLDM_MODEL_ID} | seed {args.seed} | baseline = same generator prompted with the top-1 "
              f"retrieved AudioCaps caption, unverified | ground truth = AudioCaps test recording")

    pages = [records[i:i + args.per_figure] for i in range(0, len(records), args.per_figure)]
    for number, page in enumerate(pages, 1):
        render_page(run_dir, page, run_dir / f"casv_comparison_{number}.png",
                    f"{footer} | items {(number - 1) * args.per_figure + 1}-"
                    f"{(number - 1) * args.per_figure + len(page)} of {len(records)}")


if __name__ == "__main__":
    main()
