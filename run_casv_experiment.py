# =============================================================================
# CAS-V experiment runner
#
# For every item in the manifest:
#   1. CLIP recognition + CAS-V verification of candidate sound concepts
#   2. Generate audio with the same generator and seed for two prompts:
#        baseline : unverified top-k concepts (raw CLIP similarity)
#        casv     : verified concepts
#   3. Score with CLAP against the ground-truth label text and GT audio
#
# Manifest (JSON list); "label" and "gt_audio" are optional:
#   [{"id": "busy_road", "image": "eval/images/busy_road.jpg",
#     "gt_audio": "eval/audio/busy_road.wav", "label": "busy road"}, ...]
#
# Usage:
#   py run_casv_experiment.py --manifest eval/manifest.json --run-name main
# =============================================================================

import argparse
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from casv.concepts import SCENES
from casv.verifier import CASVerifier, CLIPRecognizer, VerifierConfig, load_image, CLIP_MODEL_ID


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CAS-V image-to-audio experiment.")
    parser.add_argument("--manifest", default="eval/manifest.json")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--no-generate", action="store_true", help="Run verification only.")
    parser.add_argument("--no-clap", action="store_true", help="Skip CLAP evaluation.")
    return parser.parse_args()


def resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else (base / path)


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return (float(np.mean(values)), len(values)) if values else (None, 0)


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = VerifierConfig()

    print(f"[*] Loading CLIP ({CLIP_MODEL_ID}) on {device}")
    verifier = CASVerifier(CLIPRecognizer(device=device), config)

    generator = scorer = None
    if not args.no_generate:
        from casv.audio import AudioGenerator
        print("[*] Loading AudioLDM generator")
        generator = AudioGenerator(device=device)
    if not args.no_clap and not args.no_generate:
        from casv.audio import CLAPScorer
        print("[*] Loading CLAP scorer")
        scorer = CLAPScorer(device=device)

    records = []
    for index, item in enumerate(items, 1):
        item_id = item["id"]
        label = item.get("label")
        if label is not None and label not in SCENES:
            sys.exit(f"Unknown label '{label}' for item '{item_id}'. Valid: {sorted(SCENES)}")

        print(f"\n[{index}/{len(items)}] {item_id}")
        item_dir = run_dir / item_id
        item_dir.mkdir(exist_ok=True)

        image_path = resolve(item["image"], base)
        shutil.copy(image_path, item_dir / f"input{image_path.suffix.lower()}")
        result = verifier.verify(load_image(image_path))

        record = {
            "id": item_id,
            "image": str(image_path),
            "label": label,
            "gt_audio": str(resolve(item["gt_audio"], base)) if item.get("gt_audio") else None,
            "top_scene": result["recognition"]["top_scene"],
            "scene_confidence": result["recognition"]["scene_confidence"],
            "scene_correct": (result["recognition"]["top_scene"] == label) if label else None,
            "status": result["status"],
            "caption": result["caption"],
            "baseline_concepts": result["baseline_concepts"],
            "verified_concepts": result["verified_concepts"],
            "repaired_concepts": result["repaired_concepts"],
            "prompts": {
                "baseline": ", ".join(result["baseline_concepts"]),
                "casv": ", ".join(result["verified_concepts"]),
            },
            "metrics": {},
        }
        print(f"    scene    : {record['top_scene']} ({record['scene_confidence']:.2f}) -> {record['status']}")
        print(f"    baseline : {record['prompts']['baseline']}")
        print(f"    CAS-V    : {record['prompts']['casv']}")

        if generator is not None:
            from casv.audio import GENERATOR_SAMPLE_RATE, load_audio, save_audio

            for arm in ("baseline", "casv"):
                start = time.time()
                audio = generator.generate(record["prompts"][arm], seed=args.seed, duration=args.duration,
                                           steps=args.steps, guidance_scale=args.guidance_scale)
                save_audio(item_dir / f"{arm}.wav", audio, GENERATOR_SAMPLE_RATE)
                print(f"    generated {arm}.wav in {time.time() - start:.0f}s")

            if scorer is not None:
                gt_emb = text_emb = None
                if record["gt_audio"]:
                    gt = load_audio(record["gt_audio"], GENERATOR_SAMPLE_RATE)
                    save_audio(item_dir / "gt.wav", gt, GENERATOR_SAMPLE_RATE)
                    gt_emb = scorer.audio_embedding(gt, GENERATOR_SAMPLE_RATE)
                if label:
                    text_emb = scorer.text_embedding(SCENES[label]["caption"])
                for arm in ("baseline", "casv"):
                    emb = scorer.audio_embedding(load_audio(item_dir / f"{arm}.wav", GENERATOR_SAMPLE_RATE),
                                                 GENERATOR_SAMPLE_RATE)
                    record["metrics"][f"clap_text_{arm}"] = scorer.cosine(emb, text_emb) if text_emb is not None else None
                    record["metrics"][f"clap_gt_audio_{arm}"] = scorer.cosine(emb, gt_emb) if gt_emb is not None else None
                print("    metrics  : " + ", ".join(
                    f"{k}={v:.3f}" for k, v in record["metrics"].items() if v is not None))

        (item_dir / "record.json").write_text(json.dumps({**record, "verdicts": result["verdicts"],
                                                          "recognition": result["recognition"]}, indent=2))
        records.append(record)

    summary = {"n_items": len(records)}
    acc, n = mean_or_none([None if r["scene_correct"] is None else float(r["scene_correct"]) for r in records])
    summary["scene_accuracy"] = {"mean": acc, "n": n}
    for key in ("clap_text_baseline", "clap_text_casv", "clap_gt_audio_baseline", "clap_gt_audio_casv"):
        value, n = mean_or_none([r["metrics"].get(key) for r in records])
        summary[key] = {"mean": value, "n": n}

    from casv import audio as audio_module
    output = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": device},
        "models": {
            "clip": CLIP_MODEL_ID,
            "generator": None if args.no_generate else audio_module.AUDIOLDM_MODEL_ID,
            "clap": None if scorer is None else audio_module.CLAP_MODEL_ID,
        },
        "settings": {k: v for k, v in vars(args).items()},
        "verifier_config": asdict(config),
        "summary": summary,
        "items": records,
    }
    (run_dir / "results.json").write_text(json.dumps(output, indent=2))

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\n[+] Results written to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
