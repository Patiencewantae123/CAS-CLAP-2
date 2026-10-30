# =============================================================================
# Tune the CAS-V verifier on the AudioCaps *validation* split, then score the
# chosen configuration once on the *test* split.
#
# Objective (no audio generation needed): CLAP cosine between the prompt the
# verifier selects and the real ground-truth recording. Reported alongside:
#   baseline : top-1 retrieved caption, unverified
#   default  : VerifierConfig() defaults
#   tuned    : best configuration on val
#   oracle   : best of the k candidates per item (upper bound for selection)
#
# The grid search uses a vectorised copy of CASVerifier.decide(); the chosen
# configuration is re-checked against decide() itself on every item.
#
# Usage:
#   py -m scripts.tune_casv --val data/audiocaps/pairs/val/manifest.json \
#                           --test data/audiocaps/pairs/test/manifest.json
# =============================================================================

import argparse
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from casv import vocab
from casv.audio import CLAPScorer, GENERATOR_SAMPLE_RATE, load_audio
from casv.verifier import CASVerifier, CLIPRecognizer, VerifierConfig, load_image

WEIGHT_STEP = 0.1
ACCEPT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
ENV_CONFIDENCES = [0.6, 0.7, 0.8, 0.9, 1.01]      # 1.01 disables the environment check
SOURCE_Z = [-99.0, -1.0, -0.5, 0.0, 0.5]          # -99 disables the absent-source check


def bootstrap_ci(values, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def prepare(manifest_path, verifier, scorer):
    base = Path(manifest_path).parent
    items = []
    for item in json.loads(Path(manifest_path).read_text(encoding="utf-8")):
        rec = verifier.measure(load_image(base / item["image"]))
        gt = scorer.audio_embedding(load_audio(base / item["gt_audio"], GENERATOR_SAMPLE_RATE),
                                    GENERATOR_SAMPLE_RATE)
        captions = [c["caption"] for c in rec["candidates"]]
        fallback = f"ambient sounds of a {rec['top_scenes'][0][0]} with a {rec['top_objects'][0][0]}"
        text = scorer.text_embeddings(captions + [fallback])
        clap = (text @ gt).cpu().numpy()

        env = [vocab.caption_environment(c) for c in captions]
        min_z = [min((rec["source_z"][s] for s in vocab.mentioned_sources(c)), default=np.inf)
                 for c in captions]
        items.append({
            "id": item["id"], "rec": rec, "fallback": fallback,
            "V": np.array([c["visual"] for c in rec["candidates"]]),
            "S": np.array([c["scene_support"] for c in rec["candidates"]]),
            "O": np.array([c["object_support"] for c in rec["candidates"]]),
            "env_indoor": np.array([e == "indoor" for e in env]),
            "env_outdoor": np.array([e == "outdoor" for e in env]),
            "p_indoor": rec["env_probs"]["indoor"],
            "min_z": np.array(min_z),
            "clap": clap[:-1], "clap_fallback": float(clap[-1]),
        })
        print(f"    {item['id']}")
    return items


def select(item, cfg):
    """Vectorised CASVerifier.decide() prompt selection: index, or -1 for the fallback."""
    conflict = ((item["env_indoor"] & (1 - item["p_indoor"] >= cfg.env_conflict_confidence))
                | (item["env_outdoor"] & (item["p_indoor"] >= cfg.env_conflict_confidence))
                | (item["min_z"] < cfg.source_absent_z))
    score = cfg.w_visual * item["V"] + cfg.w_scene * item["S"] + cfg.w_object * item["O"]
    for pool in (~conflict & (score >= cfg.accept_threshold), ~conflict):
        if pool.any():
            masked = np.where(pool, score, -np.inf)
            return int(np.argmax(masked))
    return -1


def item_scores(items, cfg):
    return np.array([it["clap"][i] if (i := select(it, cfg)) >= 0 else it["clap_fallback"] for it in items])


def configs():
    steps = int(round(1 / WEIGHT_STEP))
    for a, b in itertools.product(range(steps + 1), repeat=2):
        if a + b > steps:
            continue
        w_v, w_s = a * WEIGHT_STEP, b * WEIGHT_STEP
        for acc, env, z in itertools.product(ACCEPT_THRESHOLDS, ENV_CONFIDENCES, SOURCE_Z):
            yield replace(VerifierConfig(), w_visual=round(w_v, 2), w_scene=round(w_s, 2),
                          w_object=round(1 - w_v - w_s, 2), accept_threshold=acc,
                          env_conflict_confidence=env, source_absent_z=z)


def summarise(items, cfg_default, cfg_tuned):
    baseline = np.array([it["clap"][0] for it in items])
    default = item_scores(items, cfg_default)
    tuned = item_scores(items, cfg_tuned)
    oracle = np.array([it["clap"].max() for it in items])
    out = {"n": len(items)}
    for name, values in (("baseline", baseline), ("default", default), ("tuned", tuned), ("oracle", oracle)):
        out[name] = {"mean": float(values.mean()), "ci95": bootstrap_ci(values)}
    for name, values in (("default", default), ("tuned", tuned)):
        diff = values - baseline
        out[f"{name}_minus_baseline"] = {
            "mean": float(diff.mean()), "ci95": bootstrap_ci(diff),
            "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()), "same": int((diff == 0).sum()),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Tune CAS-V on AudioCaps val, evaluate once on test.")
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", default="results/tuning")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    verifier = CASVerifier(CLIPRecognizer())
    scorer = CLAPScorer()
    print("[*] Measuring validation items")
    val = prepare(args.val, verifier, scorer)

    default = VerifierConfig()
    best_cfg, best = default, item_scores(val, default).mean()
    n_configs = 0
    for cfg in configs():
        n_configs += 1
        score = item_scores(val, cfg).mean()
        if score > best + 1e-9:
            best_cfg, best = cfg, score
    print(f"[*] Searched {n_configs} configurations; best val objective {best:.4f}")

    for it in val:
        chosen = verifier.decide(it["rec"], best_cfg)["casv_prompt"]
        index = select(it, best_cfg)
        expected = it["rec"]["candidates"][index]["caption"] if index >= 0 else it["fallback"]
        assert chosen == expected, f"vectorised selection disagrees with decide() on {it['id']}"
    print("[*] Vectorised selection matches CASVerifier.decide() on all val items")

    print("[*] Measuring test items (scored once with the chosen configuration)")
    test = prepare(args.test, verifier, scorer)

    results = {
        "objective": "CLAP cosine(selected prompt text, ground-truth recording)",
        "search_space": {"weight_step": WEIGHT_STEP, "accept_thresholds": ACCEPT_THRESHOLDS,
                         "env_conflict_confidence": ENV_CONFIDENCES, "source_absent_z": SOURCE_Z,
                         "n_configs": n_configs},
        "default_config": asdict(default),
        "tuned_config": asdict(best_cfg),
        "val": summarise(val, default, best_cfg),
        "test": summarise(test, default, best_cfg),
    }
    (out_dir / "tuning_results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "casv_tuned_config.json").write_text(json.dumps(asdict(best_cfg), indent=2))
    print(json.dumps({k: results[k] for k in ("tuned_config", "val", "test")}, indent=2))


if __name__ == "__main__":
    main()
