# =============================================================================
# Collect everything needed for the paper into journal_results/:
#   figures/  numbered figures copied from the run directories
#   tables/   CSV + LaTeX (booktabs) tables
#   RESULTS.md  results summary with every number read from result files
#
# Run after eval_scenes, train, test and the two figure
# scripts. Missing inputs are reported, never filled in.
#
# Usage:
#   py -m scripts.report
# =============================================================================

import argparse
import csv
import json
import logging
import shutil
from collections import Counter
from datetime import date
from pathlib import Path

from casv.log import setup_logging

log = logging.getLogger("casv")

OUT = Path("journal_results")


def write_table(name, header, rows, caption):
    """Write a table as CSV and LaTeX (booktabs); return it as Markdown."""
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    with open(OUT / "tables" / f"{name}.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)

    def tex(cell):
        return str(cell).replace("%", r"\%").replace("_", r"\_").replace("±", r"$\pm$")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{tex(caption)}}}",
        rf"\label{{tab:{name}}}",
        r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}",
        r"\toprule",
        " & ".join(tex(h) for h in header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(tex(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return (
        "| "
        + " | ".join(header)
        + " |\n|"
        + "---|" * len(header)
        + "\n"
        + "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    )


def ci(block):
    return f"{block['mean']:.3f} [{block['ci95'][0]:.3f}, {block['ci95'][1]:.3f}]"


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Assemble journal_results/ from experiment outputs.")
    parser.add_argument("--run", default="results/audiocaps_test400")
    parser.add_argument("--places", default="results/places365/eval.json")
    parser.add_argument("--tuning", default="results/tuning_test400/tuning_results.json")
    parser.add_argument("--categories", default="results/categories7")
    args = parser.parse_args()
    run_dir = Path(args.run)
    results_path = run_dir / "results.json"
    stats_path = run_dir / "graph_stats.json"
    n_test = json.loads(results_path.read_text())["summary"]["n_items"] if results_path.exists() else None
    # graphs/ holds charts with statistics; comparisons/<set>/ holds comparison images
    for old in ("figures", "qualitative"):  # earlier layouts
        if (OUT / old).exists():
            shutil.rmtree(OUT / old)
    for folder in ("graphs", "comparisons/fig5_categories", "comparisons/figS1_random20", "comparisons/figS2_selected"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
        for stale in (OUT / folder).glob("*.png"):
            stale.unlink()

    figures = (
        [
            (Path(args.places).with_name("fig_places365_recognition.png"), "graphs/fig1_scene_recognition.png"),
            (Path(args.tuning).with_name("fig_prompt_selection.png"), "graphs/fig2_prompt_selection.png"),
            (run_dir / "fig_clap_comparison.png", "graphs/fig3_generated_audio_clap.png"),
            (run_dir / "fig_verification_stats.png", "graphs/fig4_verification_statistics.png"),
            (
                Path(args.categories) / "comparisons" / "casv_comparison_1.png",
                "comparisons/fig5_categories/fig5_categories.png",
            ),
        ]
        + [
            (p, f"comparisons/figS1_random20/figS1{chr(ord('a') + i)}_random20.png")
            for i, p in enumerate(sorted((run_dir / "comparisons").glob("casv_comparison_*.png")))
        ]
        + [
            (
                run_dir / "comparisons_selected" / f"casv_comparison_{i}.png",
                f"comparisons/figS2_selected/figS2_page{i}.png",
            )
            for i in range(1, len(list((run_dir / "comparisons_selected").glob("casv_comparison_*.png"))) + 1)
        ]
    )
    missing = []
    for src, dest in figures:
        if src.exists():
            shutil.copy(src, OUT / dest)
            caption = src.with_name(src.stem + "_caption.txt")
            if caption.exists():
                shutil.copy(caption, (OUT / dest).with_name(Path(dest).stem + "_caption.txt"))
        else:
            missing.append(str(src))

    sections = []

    if Path(args.places).exists():
        p = json.loads(Path(args.places).read_text())["summary"]
        sections.append(
            "## Table 1. Scene recognition on Places365 val\n\n"
            + write_table(
                "table1_scene_recognition",
                ["Metric", "Value", "n"],
                [
                    ["Scene top-1 accuracy (365-way)", f"{p['scene_top1']:.1%}", p["n_images"]],
                    ["Scene top-5 accuracy", f"{p['scene_top5']:.1%}", p["n_images"]],
                    ["Indoor/outdoor accuracy", f"{p['env_accuracy']:.1%}", p["n_images"]],
                    [
                        "Indoor/outdoor accuracy, p >= 0.7",
                        f"{p['env_accuracy_when_p_ge_0.7']:.1%}",
                        round(p["env_coverage_p_ge_0.7"] * p["n_images"]),
                    ],
                ],
                f"CLIP ({p['clip_model']}) zero-shot recognition on a stratified sample of Places365 val "
                "(10 images per class).",
            )
        )
    else:
        missing.append(args.places)

    if Path(args.tuning).exists():
        t = json.loads(Path(args.tuning).read_text())
        rows = []
        for split in ("val", "test"):
            for method in ("baseline", "default", "tuned", "oracle"):
                rows.append([split, method, ci(t[split][method]), t[split]["n"]])
        sections.append(
            "## Table 2. Prompt selection: CLAP(prompt, ground-truth recording)\n\n"
            + write_table(
                "table2_prompt_selection",
                ["Split", "Method", "Mean [95% CI]", "n"],
                rows,
                "CLAP cosine between the selected prompt and the real AudioCaps recording. "
                "Tuned on val only; test is never used for tuning.",
            )
        )
        lines = []
        for name in ("default", "tuned"):
            d = t["test"][f"{name}_minus_baseline"]
            lines.append(
                f"- Test, {name} − baseline: {ci(d)}; better on {d['better']}, worse on {d['worse']}, "
                f"unchanged on {d['same']} items."
            )
        tuned = t["tuned_config"]
        checks_off = tuned["env_conflict_confidence"] > 1 and tuned["source_absent_z"] < -50
        lines.append(f"- Tuned configuration: `{json.dumps(tuned)}`")
        lines.append(
            "- The primary CAS-V configuration is the default, fixed before any data was seen; "
            "the tuned configuration is an ablation."
        )
        if checks_off:
            lines.append(
                "- The val-optimal configuration disables both explicit conflict checks: under this "
                "objective, rejecting sounds that are not visible does not increase agreement with the "
                "recording, which itself contains off-screen sounds (narration, music, speech)."
            )
        oracle_gap = t["test"]["oracle"]["mean"] - t["test"]["baseline"]["mean"]
        lines.append(f"- Selection headroom on test: oracle − baseline = {oracle_gap:+.3f}.")
        sections.append("\n".join(lines))
    else:
        missing.append(args.tuning)

    categories_path = Path(args.categories) / "results.json"
    if categories_path.exists():
        cat = json.loads(categories_path.read_text())
        rows = []
        for r in cat["items"]:
            m = r["metrics"]
            rows.append(
                [
                    r["label"],
                    r["id"],
                    "yes" if r["prompts"]["baseline"] != r["prompts"]["casv"] else "no",
                    f"{m['clap_gt_audio_baseline']:.3f}",
                    f"{m['clap_gt_audio_casv']:.3f}",
                    f"{m['clap_gt_audio_casv'] - m['clap_gt_audio_baseline']:+.3f}",
                ]
            )
        s = cat["summary"]["clap_gt_audio"]
        rows.append(
            [
                "Mean",
                "",
                f"{cat['summary']['prompt_changed']}/{cat['summary']['n_items']}",
                f"{s['baseline_mean']:.3f}",
                f"{s['casv_mean']:.3f}",
                f"{s['mean_diff']:+.3f}",
            ]
        )
        sections.append(
            "## Table 5. Figure 5 examples: CLAP(generated audio, GT recording)\n\n"
            + write_table(
                "table5_figure5_examples",
                ["Category", "AudioCaps id", "Prompt changed", "Baseline", "CAS-V", "Δ"],
                rows,
                "Scores for the seven Figure 5 columns. Seven examples are illustrative, not a benchmark; "
                "see Table 3 for the full test result.",
            )
        )
    else:
        missing.append(str(categories_path))

    if results_path.exists() and stats_path.exists():
        res = json.loads(results_path.read_text())
        stats = json.loads(stats_path.read_text())
        rows = []
        for group, suffix in (("clap", ""), ("clap_new_items", ", new items only")):
            for key, label in (
                ("clap_text", "CLAP(audio, GT captions)"),
                ("clap_gt_audio", "CLAP(audio, GT recording)"),
            ):
                if key in stats.get(group, {}):
                    c = stats[group][key]
                    p_value = "n/a" if c["wilcoxon_p_changed_items"] is None else f"{c['wilcoxon_p_changed_items']:.3g}"
                    rows.append(
                        [
                            label + suffix,
                            f"{c['baseline_mean']:.3f}",
                            f"{c['casv_mean']:.3f}",
                            f"{c['mean_diff']:+.3f} [{c['mean_diff_ci95'][0]:+.3f}, {c['mean_diff_ci95'][1]:+.3f}]",
                            f"{c['casv_better']}/{c['casv_worse']}",
                            p_value,
                            c["n"],
                        ]
                    )
        sections.append(
            "## Table 3. Generated audio (AudioLDM-S), baseline vs CAS-V\n\n"
            + write_table(
                "table3_generated_audio",
                ["Metric", "Baseline", "CAS-V", "Δ [95% CI]", "better/worse", "Wilcoxon p", "n"],
                rows,
                f"Same generator ({res['models']['generator']}) and seed for both arms; "
                f"CAS-V changed the prompt on {res['summary']['prompt_changed']} of {res['summary']['n_items']} items.",
            )
        )
        if "clap_new_items" in stats:
            earlier = "; ".join(
                f"{e['run']}: {e['n_shared']} of its {e['n_items']} items are in this sample, "
                f"{'same' if e['same_config'] else 'DIFFERENT'} verifier configuration"
                for e in stats["earlier_runs"]
            )
            n_new = next(iter(stats["clap_new_items"].values()))["n"]
            sections.append(
                f"- The primary result uses all {res['summary']['n_items']} items, drawn in the same seeded clip "
                "order as the earlier test results; an earlier clip is absent here if its download failed. "
                f"The new-items rows are a sensitivity analysis on the {n_new} clips in no earlier run ({earlier})."
            )
        v = stats["verification"]
        conflicts = Counter(v["conflicts"]).most_common(8)
        sections.append(
            "## Table 4. Verification statistics\n\n"
            + write_table(
                "table4_verification",
                ["Quantity", "Count"],
                [[f"Candidates {k}", v["candidate_decisions"].get(k, 0)] for k in ("ACCEPT", "REPAIR", "REJECT")]
                + [[f"Images {k}", v["item_status"].get(k, 0)] for k in ("ACCEPT", "REPAIR", "UNCERTAIN")]
                + [[f"Conflict: {name}", n] for name, n in conflicts],
                "Decisions over all retrieved candidate captions and final per-image status.",
            )
        )
    else:
        missing += [str(p) for p in (results_path, stats_path) if not p.exists()]

    header = f"""# CAS-V results

Generated {date.today().isoformat()} by `scripts/report.py`. Every number below is read
from result files in `results/`; nothing is entered by hand.

## Setup

| Component | Choice |
|---|---|
| Recognition | CLIP ViT-B/32 zero-shot: Places365 scenes (365), ImageNet objects (1000) |
| Sound candidates | AudioCaps **train** captions (test/val captions never enter the candidate pool) |
| Verification | visual + scene + object support; explicit conflicts: indoor/outdoor, absent sound source |
| Generator | AudioLDM-S (`cvssp/audioldm-s-full-v2`), same seed for both arms |
| Evaluation | AudioCaps test clips: video frame as input, real recording + 5 human captions as ground truth; CLAP (`laion/clap-htsat-unfused`), clips over 10 s cropped to their first 10 s |
| Baseline | same generator prompted with the top-1 retrieved caption, unverified |
| Tuning | grid search on AudioCaps **val**; test is never used for tuning |

## Figures

| File | Content |
|---|---|
| `graphs/fig1_scene_recognition.png` | Places365 accuracy and confidence calibration |
| `graphs/fig2_prompt_selection.png` | Prompt selection on val/test with 95% CIs |
| `graphs/fig3_generated_audio_clap.png` | Generated audio: baseline vs CAS-V, paired |
| `graphs/fig4_verification_statistics.png` | Candidate decisions and conflict types |
| `comparisons/fig5_categories/` | One AudioCaps test clip per category: first clip in seeded order whose captions match the category rule and whose frame passed visual screening (frames only, before generation). Rejections and the one relabel (forest → field) are listed in `data/audiocaps/pairs/categories/selection_rule.json` |
| `comparisons/figS1_random20/` | Supplementary: first 20 items of the random test sample (not selected) |
| `comparisons/figS2_selected/` | Illustrations, not results: clear examples of named sound categories where CAS-V's verified caption names the sound, so they show what the method does when it works. No CLAP score was used; the rule and its counts are in each page's caption and in `selected_clips.json` next to the run |

LaTeX versions of all tables are in `tables/`.
"""
    sample_note = (
        f"- Results rest on {n_test} test pairs; report confidence intervals with every mean.\n" if n_test else ""
    )
    limitations = f"""
## Limitations

{sample_note}- CLAP similarity is an automatic proxy; no listening test has been run.
- CLAP embeddings of a recording change when it is shifted by a few milliseconds, so single-item scores
  are noisy; compare means with their confidence intervals, not individual items.
- AudioCaps frames are single video frames; off-screen sounds (narration, music) in the ground truth
  cannot be recovered from the image by any method.
- Places365/ImageNet are used for zero-shot vocabularies and Places365 val for recognition accuracy only;
  they contain no audio.
"""
    if missing:
        limitations += "\n## Missing inputs at build time\n\n" + "\n".join(f"- `{m}`" for m in missing) + "\n"
    (OUT / "RESULTS.md").write_text(header + "\n" + "\n\n".join(sections) + "\n" + limitations, encoding="utf-8")
    log.info(f"{OUT.resolve()}")
    for m in missing:
        log.warning(f"missing input: {m}")


if __name__ == "__main__":
    main()
