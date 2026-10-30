# =============================================================================
# Evaluate CAS-V's scene recognition on the Places365 validation set.
#
# Measures the two signals the verifier depends on:
#   - scene top-1 / top-5 accuracy (365-way CLIP zero-shot)
#   - indoor/outdoor accuracy of P(indoor), used for environment conflicts
# Per-image confidences are saved so calibration can be plotted.
#
# Usage:
#   py -m scripts.eval_places365 --per-class 10
# =============================================================================

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from casv import vocab
from casv.verifier import CLIPRecognizer, load_image, CLIP_MODEL_ID

PLACES_DIR = vocab.DATA_DIR / "places365"


def main():
    parser = argparse.ArgumentParser(description="CAS-V scene recognition on Places365 val.")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/places365/eval.json")
    args = parser.parse_args()

    names, envs = vocab.load_places365()
    by_class = defaultdict(list)
    for line in (PLACES_DIR / "filelists" / "places365_val.txt").read_text().split("\n"):
        if line.strip():
            filename, label = line.split()
            by_class[int(label)].append(filename)
    rng = random.Random(args.seed)
    sample = [(f, label) for label in sorted(by_class)
              for f in rng.sample(sorted(by_class[label]), min(args.per_class, len(by_class[label])))]
    print(f"[*] {len(sample)} images ({args.per_class} per class, seed {args.seed})")

    recognizer = CLIPRecognizer()
    scene_emb = recognizer.encode_texts([f"a photo of a {n}" for n in names])
    indoor_mask = np.array([e == "indoor" for e in envs])

    rows = []
    for i, (filename, label) in enumerate(sample, 1):
        probs = recognizer.softmax(scene_emb @ recognizer.encode_image(load_image(PLACES_DIR / "val_256" / filename)))
        top5 = np.argsort(probs)[::-1][:5]
        p_indoor = float(probs[indoor_mask].sum())
        rows.append({
            "file": filename,
            "label": names[label],
            "pred": names[top5[0]],
            "confidence": float(probs[top5[0]]),
            "top1": bool(top5[0] == label),
            "top5": bool(label in top5),
            "env_label": envs[label],
            "p_indoor": p_indoor,
            "env_correct": (p_indoor >= 0.5) == (envs[label] == "indoor"),
        })
        if i % 500 == 0:
            print(f"    {i}/{len(sample)}")

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    confident = [r for r in rows if max(r["p_indoor"], 1 - r["p_indoor"]) >= 0.7]
    summary = {
        "n_images": len(rows),
        "clip_model": CLIP_MODEL_ID,
        "scene_top1": mean("top1"),
        "scene_top5": mean("top5"),
        "env_accuracy": mean("env_correct"),
        "env_accuracy_when_p_ge_0.7": float(np.mean([r["env_correct"] for r in confident])),
        "env_coverage_p_ge_0.7": len(confident) / len(rows),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "images": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
