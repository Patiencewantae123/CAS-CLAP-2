# =============================================================================
# CAS-V experiment runner
#
# For every item in the manifest:
#   1. CAS-V recognition (Places365 / ImageNet) and verification of
#      AudioCaps-train caption candidates
#   2. Generate audio with the same generator and seed for two prompts:
#        baseline : top-1 retrieved caption, unverified
#        casv     : verified caption
#   3. Score with CLAP against the ground-truth captions and GT audio
#
# Manifest (JSON list); "captions" and "gt_audio" are optional:
#   [{"id": "7fmOlUlwoNg", "image": "frames/7fmOlUlwoNg.jpg",
#     "gt_audio": "audio/7fmOlUlwoNg.wav", "captions": ["...", "..."]}, ...]
#
# Usage:
#   py run_casv_experiment.py --manifest data/audiocaps/test_subset/manifest.json --run-name main
# =============================================================================

import argparse
import json
import platform
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from casv.verifier import CASVerifier, CLIPRecognizer, VerifierConfig, load_image, CLIP_MODEL_ID

ARMS = ("baseline", "casv")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CAS-V image-to-audio experiment.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N items.")
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


def summarise(records):
    summary = {"n_items": len(records),
               "status_counts": {s: sum(r["status"] == s for r in records)
                                 for s in ("ACCEPT", "REPAIR", "UNCERTAIN")},
               "prompt_changed": sum(r["prompts"]["baseline"] != r["prompts"]["casv"] for r in records)}
    for metric in ("clap_text", "clap_gt_audio"):
        pairs = [(r["metrics"].get(f"{metric}_baseline"), r["metrics"].get(f"{metric}_casv")) for r in records]
        pairs = [(b, c) for b, c in pairs if b is not None and c is not None]
        if not pairs:
            continue
        base, casv = np.array(pairs).T
        diff = casv - base
        summary[metric] = {
            "n": len(pairs),
            "baseline_mean": float(base.mean()),
            "casv_mean": float(casv.mean()),
            "mean_diff": float(diff.mean()),
            "diff_std": float(diff.std(ddof=1)) if len(pairs) > 1 else None,
            "casv_wins": int((diff > 0).sum()),
        }
    return summary


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    items = json.loads(manifest_path.read_text(encoding="utf-8"))[:args.limit]
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
        if not args.no_clap:
            from casv.audio import CLAPScorer
            print("[*] Loading CLAP scorer")
            scorer = CLAPScorer(device=device)

    records = []
    for index, item in enumerate(items, 1):
        item_id = item["id"]
        print(f"\n[{index}/{len(items)}] {item_id}")
        item_dir = run_dir / item_id
        item_dir.mkdir(exist_ok=True)

        image_path = resolve(item["image"], base)
        shutil.copy(image_path, item_dir / f"input{image_path.suffix.lower()}")
        result = verifier.verify(load_image(image_path))

        record = {
            "id": item_id,
            "image": str(image_path),
            "gt_audio": str(resolve(item["gt_audio"], base)) if item.get("gt_audio") else None,
            "gt_captions": item.get("captions", []),
            "top_scene": result["top_scenes"][0],
            "top_object": result["top_objects"][0],
            "env_probs": result["env_probs"],
            "status": result["status"],
            "n_rejected": result["n_rejected"],
            "prompts": {"baseline": result["baseline_prompt"], "casv": result["casv_prompt"]},
            "metrics": {},
        }
        print(f"    scene    : {record['top_scene'][0]} ({record['top_scene'][1]:.2f}), "
              f"object: {record['top_object'][0]} -> {record['status']} "
              f"({record['n_rejected']}/{len(result['verdicts'])} candidates rejected)")
        print(f"    baseline : {record['prompts']['baseline']}")
        print(f"    CAS-V    : {record['prompts']['casv']}")

        if generator is not None:
            from casv.audio import GENERATOR_SAMPLE_RATE, load_audio, save_audio

            for arm in ARMS:
                if arm == "casv" and record["prompts"]["casv"] == record["prompts"]["baseline"]:
                    # Same prompt and seed give identical audio; reuse it.
                    shutil.copy(item_dir / "baseline.wav", item_dir / "casv.wav")
                    continue
                start = time.time()
                audio = generator.generate(record["prompts"][arm], seed=args.seed, duration=args.duration,
                                           steps=args.steps, guidance_scale=args.guidance_scale)
                save_audio(item_dir / f"{arm}.wav", audio, GENERATOR_SAMPLE_RATE)
                print(f"    generated {arm}.wav in {time.time() - start:.0f}s")

            if record["gt_audio"]:
                save_audio(item_dir / "gt.wav", load_audio(record["gt_audio"], GENERATOR_SAMPLE_RATE),
                           GENERATOR_SAMPLE_RATE)

            if scorer is not None:
                gt_emb = text_emb = None
                if record["gt_audio"]:
                    gt_emb = scorer.audio_embedding(load_audio(item_dir / "gt.wav", GENERATOR_SAMPLE_RATE),
                                                    GENERATOR_SAMPLE_RATE)
                if record["gt_captions"]:
                    text_emb = torch.nn.functional.normalize(
                        torch.stack([scorer.text_embedding(c) for c in record["gt_captions"]]).mean(0), dim=-1)
                for arm in ARMS:
                    emb = scorer.audio_embedding(load_audio(item_dir / f"{arm}.wav", GENERATOR_SAMPLE_RATE),
                                                 GENERATOR_SAMPLE_RATE)
                    record["metrics"][f"clap_text_{arm}"] = scorer.cosine(emb, text_emb) if text_emb is not None else None
                    record["metrics"][f"clap_gt_audio_{arm}"] = scorer.cosine(emb, gt_emb) if gt_emb is not None else None
                print("    metrics  : " + ", ".join(
                    f"{k}={v:.3f}" for k, v in record["metrics"].items() if v is not None))

        (item_dir / "record.json").write_text(json.dumps({**record, "verdicts": result["verdicts"],
                                                          "top_scenes": result["top_scenes"],
                                                          "top_objects": result["top_objects"]}, indent=2))
        records.append(record)

    from casv import audio as audio_module
    summary = summarise(records)
    output = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": device},
        "models": {
            "clip": CLIP_MODEL_ID,
            "generator": None if args.no_generate else audio_module.AUDIOLDM_MODEL_ID,
            "clap": None if scorer is None else audio_module.CLAP_MODEL_ID,
        },
        "vocabulary": {"scenes": "Places365", "objects": "ImageNet-1k", "sound_candidates": "AudioCaps train captions"},
        "settings": vars(args),
        "verifier_config": asdict(config),
        "summary": summary,
        "items": records,
    }
    (run_dir / "results.json").write_text(json.dumps(output, indent=2))

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\n[+] Results written to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
