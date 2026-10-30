# =============================================================================
# TRAIN: fit the CAS-V verifier on the AudioCaps *validation* split, then score
# the chosen configuration once on the *test* split.
#
# What is trained: only the verifier's parameters (three evidence weights, the
# accept threshold and the two conflict thresholds) by grid search. No neural
# network weights are trained; CLIP, AudioLDM and CLAP are used pretrained and
# frozen. The result is saved to results/tuning/casv_tuned_config.json.
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
#   py -m scripts.train --val data/audiocaps/pairs/val/manifest.json \
#                           --test data/audiocaps/pairs/test400/manifest.json \
#                           --out results/tuning_test400
# =============================================================================

import argparse
import itertools
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from casv import vocab
from casv.audio import GENERATOR_SAMPLE_RATE, CLAPScorer, load_audio
from casv.log import setup_logging
from casv.stats import bootstrap_ci
from casv.verifier import CASVerifier, CLIPRecognizer, VerifierConfig, load_image

log = logging.getLogger("casv")

WEIGHT_STEP = 0.1
ACCEPT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
ENV_CONFIDENCES = [0.6, 0.7, 0.8, 0.9, 1.01]  # 1.01 disables the environment check
SOURCE_Z = [-99.0, -1.0, -0.5, 0.0, 0.5]  # -99 disables the absent-source check


def fallback_prompt(rec: dict) -> str:
    """Prompt CASVerifier.decide() uses when every candidate conflicts."""
    return f"ambient sounds of a {rec['top_scenes'][0][0]} with a {rec['top_objects'][0][0]}"


def candidate_arrays(rec: dict) -> dict:
    """Config-independent arrays needed by select(), built from a CASVerifier.measure() record."""
    captions = [c["caption"] for c in rec["candidates"]]
    env = [vocab.caption_environment(c) for c in captions]
    min_z = [min((rec["source_z"][s] for s in vocab.mentioned_sources(c)), default=np.inf) for c in captions]
    return {
        "rec": rec,
        "fallback": fallback_prompt(rec),
        "V": np.array([c["visual"] for c in rec["candidates"]]),
        "S": np.array([c["scene_support"] for c in rec["candidates"]]),
        "O": np.array([c["object_support"] for c in rec["candidates"]]),
        "env_indoor": np.array([e == "indoor" for e in env]),
        "env_outdoor": np.array([e == "outdoor" for e in env]),
        "p_indoor": rec["env_probs"]["indoor"],
        "min_z": np.array(min_z),
    }


def prepare(manifest_path, verifier, scorer):
    """Measure every manifest item once and attach CLAP scores of each candidate vs the GT recording."""
    base = Path(manifest_path).parent
    items = []
    for item in json.loads(Path(manifest_path).read_text(encoding="utf-8")):
        rec = verifier.measure(load_image(base / item["image"]))
        gt = scorer.audio_embedding(load_audio(base / item["gt_audio"], GENERATOR_SAMPLE_RATE), GENERATOR_SAMPLE_RATE)
        arrays = candidate_arrays(rec)
        captions = [c["caption"] for c in rec["candidates"]]
        clap = (scorer.text_embeddings(captions + [arrays["fallback"]]) @ gt).cpu().numpy()
        items.append({"id": item["id"], **arrays, "clap": clap[:-1], "clap_fallback": float(clap[-1])})
        log.info(f"    {item['id']}")
    return items


def select(item, cfg):
    """Vectorised CASVerifier.decide() prompt selection: index, or -1 for the fallback."""
    conflict = (
        (item["env_indoor"] & (1 - item["p_indoor"] >= cfg.env_conflict_confidence))
        | (item["env_outdoor"] & (item["p_indoor"] >= cfg.env_conflict_confidence))
        | (item["min_z"] < cfg.source_absent_z)
    )
    score = cfg.w_visual * item["V"] + cfg.w_scene * item["S"] + cfg.w_object * item["O"]
    for pool in (~conflict & (score >= cfg.accept_threshold), ~conflict):
        if pool.any():
            masked = np.where(pool, score, -np.inf)
            return int(np.argmax(masked))
    return -1


def item_scores(items, cfg):
    """Objective value per item for one configuration."""
    return np.array([it["clap"][i] if (i := select(it, cfg)) >= 0 else it["clap_fallback"] for it in items])


def configs():
    """Every configuration in the search grid."""
    steps = int(round(1 / WEIGHT_STEP))
    for a, b in itertools.product(range(steps + 1), repeat=2):
        if a + b > steps:
            continue
        w_v, w_s = a * WEIGHT_STEP, b * WEIGHT_STEP
        for acc, env, z in itertools.product(ACCEPT_THRESHOLDS, ENV_CONFIDENCES, SOURCE_Z):
            yield replace(
                VerifierConfig(),
                w_visual=round(w_v, 2),
                w_scene=round(w_s, 2),
                w_object=round(1 - w_v - w_s, 2),
                accept_threshold=acc,
                env_conflict_confidence=env,
                source_absent_z=z,
            )


def summarise(items, cfg_default, cfg_tuned):
    """Baseline, default, tuned and oracle means with CIs and paired differences."""
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
            "mean": float(diff.mean()),
            "ci95": bootstrap_ci(diff),
            "better": int((diff > 0).sum()),
            "worse": int((diff < 0).sum()),
            "same": int((diff == 0).sum()),
        }
    return out


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Tune CAS-V on AudioCaps val, evaluate once on test.")
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", default="results/tuning")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    verifier = CASVerifier(CLIPRecognizer())
    scorer = CLAPScorer()
    log.info("Measuring validation items")
    val = prepare(args.val, verifier, scorer)

    default = VerifierConfig()
    best_cfg, best = default, item_scores(val, default).mean()
    n_configs = 0
    for cfg in configs():
        n_configs += 1
        score = item_scores(val, cfg).mean()
        if score > best + 1e-9:
            best_cfg, best = cfg, score
    log.info(f"Searched {n_configs} configurations; best val objective {best:.4f}")

    for it in val:
        chosen = verifier.decide(it["rec"], best_cfg)["casv_prompt"]
        index = select(it, best_cfg)
        expected = it["rec"]["candidates"][index]["caption"] if index >= 0 else it["fallback"]
        assert chosen == expected, f"vectorised selection disagrees with decide() on {it['id']}"
    log.info("Vectorised selection matches CASVerifier.decide() on all val items")

    log.info("Measuring test items (scored once with the chosen configuration)")
    test = prepare(args.test, verifier, scorer)

    results = {
        "objective": "CLAP cosine(selected prompt text, ground-truth recording)",
        "search_space": {
            "weight_step": WEIGHT_STEP,
            "accept_thresholds": ACCEPT_THRESHOLDS,
            "env_conflict_confidence": ENV_CONFIDENCES,
            "source_absent_z": SOURCE_Z,
            "n_configs": n_configs,
        },
        "default_config": asdict(default),
        "tuned_config": asdict(best_cfg),
        "val": summarise(val, default, best_cfg),
        "test": summarise(test, default, best_cfg),
    }
    (out_dir / "tuning_results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "casv_tuned_config.json").write_text(json.dumps(asdict(best_cfg), indent=2))
    log.info(json.dumps({k: results[k] for k in ("tuned_config", "val", "test")}, indent=2))


if __name__ == "__main__":
    main()
