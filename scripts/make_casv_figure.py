# =============================================================================
# Qualitative comparison figure for a CAS-V run (layout of docs/target_figure.png).
#
# Rows: Input Image | Existing Research (Without CAS-V) |
#       Ours (CAS-V): Semantic Text (Verified Concepts), Spectrogram (Generated
#       Audio), Caption (Sound Type) | Target Sound (Ground Truth)
#
# Every panel is rendered from files produced by run_casv_experiment.py; the
# concepts and sound type are derived from the verified caption only. A
# missing ground-truth recording is shown as an empty panel, never filled in.
# Provenance (clip ids, prompts, recognition) goes to a caption .txt file.
#
# Usage:
#   py -m scripts.make_casv_figure --run results/audiocaps_test100 [--items <ytid>,<ytid>,...]
# =============================================================================

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from casv.audio import GENERATOR_SAMPLE_RATE, load_audio, mel_spectrogram_db

RED, BLUE, INK = "#c0392b", "#1f3a93", "#111111"
ROWS = [  # (label, colour, height)
    ("Input Image", INK, 1.15),
    ("Existing Research\n(Without CAS-V)", RED, 0.95),
    ("Semantic Text\n(Verified Concepts)", BLUE, 0.62),
    ("Spectrogram\n(Generated Audio)", BLUE, 0.95),
    ("Caption\n(Sound Type)", BLUE, 0.42),
    ("Target Sound\n(Ground Truth)", INK, 0.95),
]
OURS_ROWS = (2, 4)
DB_RANGE = 80           # every panel: own peak, 80 dB below it
GT_WINDOW = (2.5, 7.5)  # 5 s of the 10 s recording, centred on the input frame (t = 5 s)
CLAUSE_SPLIT = r",|;|\band then\b|\bthen\b|\bfollowed by\b|\bwhile\b|\bas\b|\band\b|\balong with\b|\bwith\b"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})


LEADING = r"^(a|an|the|some|several|multiple|series of|a series of|sounds? of|noises? of)\s+"
TRAILING_AUX = r"\s+(is|are|was|were|being|been|occurs|occur)(\s+.*)?$"
STOP_END = {"on", "of", "in", "by", "to", "its", "the", "a", "an", "is", "are", "at", "into", "from"}
NOT_A_SOURCE = {"one", "another", "it", "they", "this", "that", "low", "muffled", "continuous", "loud",
                "soft", "briefly", "repeatedly", "background", "distance"}


def clean_phrase(phrase):
    phrase = phrase.strip(" .").lower()
    for _ in range(2):
        phrase = re.sub(LEADING, "", phrase)
    return re.sub(TRAILING_AUX, "", phrase).strip()


def verified_concepts(caption, limit=4):
    parts = [clean_phrase(p) for p in re.split(CLAUSE_SPLIT, caption, flags=re.IGNORECASE)]
    parts = [p for p in parts if p and p.split()[0] not in NOT_A_SOURCE]
    return list(dict.fromkeys(parts))[:limit]


def short_head(concept, max_words=3, keep_whole=5):
    if len(concept.split()) <= keep_whole:
        return concept
    words = concept.split()[:max_words]
    while words and words[-1] in STOP_END:
        words.pop()
    return " ".join(words)


def sound_type(concepts):
    heads = []
    for concept in concepts:
        head = short_head(concept)
        if head and head not in heads:
            heads.append(head)
        if len(heads) == 2:
            break
    return " / ".join(heads).capitalize()


def draw_spectrogram(ax, wav_path, window=None):
    """window=(start_s, end_s) crops the audio before the spectrogram is computed."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if not wav_path.exists():
        ax.set_facecolor("#eeeeee")
        ax.text(0.5, 0.5, "not provided", ha="center", va="center", fontsize=8, color="#777777",
                transform=ax.transAxes)
        return
    audio = load_audio(wav_path, GENERATOR_SAMPLE_RATE)
    if window:
        audio = audio[int(window[0] * GENERATOR_SAMPLE_RATE):int(window[1] * GENERATOR_SAMPLE_RATE)]
    mel = mel_spectrogram_db(audio)
    ax.imshow(mel, origin="lower", aspect="auto", cmap="inferno", vmin=mel.max() - DB_RANGE, vmax=mel.max())


def shorten(text, width, max_lines):
    return textwrap.fill(text, width, max_lines=max_lines, placeholder=" ...")


def draw_text_box(ax, text, face, edge, fontsize):
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.03, 0.06), 0.94, 0.88, boxstyle="round,pad=0.01,rounding_size=0.04",
                                transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=0.8))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=fontsize, color=BLUE,
            transform=ax.transAxes, linespacing=1.25)


def load_records(run_dir):
    # Read per-item records directly so figures can be made while a run is in
    # progress. Items are kept in processing order (= manifest order).
    records = []
    for path in sorted(run_dir.glob("*/record.json"), key=lambda p: p.stat().st_mtime):
        if (path.parent / "casv.wav").exists():
            records.append(json.loads(path.read_text()))
    return records


def render_page(run_dir, items, output, provenance):
    n_cols = len(items)
    fig = plt.figure(figsize=(1.5 + 1.75 * n_cols, 8.6), dpi=250)
    grid = fig.add_gridspec(len(ROWS), n_cols + 1, width_ratios=[1.05] + [1] * n_cols,
                            height_ratios=[h for _, _, h in ROWS], wspace=0.07, hspace=0.16)

    label_axes = []
    for row, (label, color, _) in enumerate(ROWS):
        ax = fig.add_subplot(grid[row, 0])
        ax.axis("off")
        y = 0.32 if row == OURS_ROWS[0] else 0.5
        ax.text(0.5, y, label, ha="center", va="center", fontsize=8, fontweight="bold", color=color,
                transform=ax.transAxes)
        label_axes.append(ax)
    label_axes[OURS_ROWS[0]].text(0.5, 0.88, "Ours (CAS-V)", ha="center", va="center", fontsize=10,
                                  fontweight="bold", color=BLUE, transform=label_axes[OURS_ROWS[0]].transAxes)

    lines = []
    for col, record in enumerate(items, start=1):
        item_dir = run_dir / record["id"]
        title = (record.get("label") or record["top_scene"][0]).title()
        concepts = verified_concepts(record["prompts"]["casv"])

        ax = fig.add_subplot(grid[0, col])
        ax.imshow(Image.open(next(item_dir.glob("input.*"))).convert("RGB").resize((320, 240)))
        ax.set_title(textwrap.fill(title, 18), fontsize=9.5, fontweight="bold", pad=4)
        ax.axis("off")

        for row, wav, window in ((1, "baseline.wav", None), (3, "casv.wav", None), (5, "gt.wav", GT_WINDOW)):
            draw_spectrogram(fig.add_subplot(grid[row, col]), item_dir / wav, window)
        gt_caption = record["gt_captions"][0] if record["gt_captions"] else ""
        fig.axes[-1].set_xlabel(shorten(gt_caption, 30, 3), fontsize=6.5, labelpad=3)

        draw_text_box(fig.add_subplot(grid[2, col]), shorten(", ".join(concepts), 26, 4),
                      "#edf5ea", "#8fbf7f", 6.3)
        draw_text_box(fig.add_subplot(grid[4, col]), shorten(sound_type(concepts), 24, 2),
                      "#eaf0fb", "#8aa6d6", 6.8)

        lines.append(f"({col}) {title} — AudioCaps {record['id']}; recognised scene/object: "
                     f"{record['top_scene'][0]} / {record['top_object'][0]} ({record['status']}, "
                     f"{record['n_rejected']} of 20 candidates rejected); baseline prompt: "
                     f"\"{record['prompts']['baseline']}\"; CAS-V prompt: \"{record['prompts']['casv']}\"; "
                     f"ground-truth captions: " + " | ".join(record["gt_captions"]))

    top = label_axes[OURS_ROWS[0]].get_position()
    bottom = label_axes[OURS_ROWS[1]].get_position()
    right = fig.axes[-1].get_position().x1
    fig.patches.append(FancyBboxPatch((top.x0 - 0.004, bottom.y0 - 0.010), right - top.x0 + 0.012,
                                      top.y1 - bottom.y0 + 0.018, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fill=False, edgecolor=BLUE, linewidth=1.0,
                                      zorder=0))

    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    output.with_name(output.stem + "_caption.txt").write_text(provenance + "\n\n" + "\n".join(lines) + "\n",
                                                               encoding="utf-8")
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
    provenance = (f"Qualitative comparison on AudioCaps test clips (run '{run_dir.name}'). Input: one video "
                  f"frame. Existing research: {AUDIOLDM_MODEL_ID} prompted with the top-1 retrieved AudioCaps "
                  f"caption, unverified. Ours: the same generator and seed ({args.seed}) prompted with the "
                  f"CAS-V-verified caption; verified concepts and sound type are phrases of that caption. "
                  f"Recognition: {CLIP_MODEL_ID} zero-shot over Places365 scenes and ImageNet objects. "
                  f"Target sound: the real AudioCaps recording (5 s window centred on the input frame; metrics "
                  f"use the full 10 s) with its first human caption. Generated clips are 5 s. Mel "
                  f"spectrograms, 16 kHz, 64 bands, each scaled to its own peak with an 80 dB range.")

    pages = [records[i:i + args.per_figure] for i in range(0, len(records), args.per_figure)]
    for number, page in enumerate(pages, 1):
        render_page(run_dir, page, run_dir / f"casv_comparison_{number}.png",
                    f"{provenance} Items {(number - 1) * args.per_figure + 1}-"
                    f"{(number - 1) * args.per_figure + len(page)} of {len(records)}.")


if __name__ == "__main__":
    main()
