# =============================================================================
# Collect everything needed for the paper into journal_results/:
#   figures/  numbered figures copied from the run directories
#   tables/   CSV + LaTeX (booktabs) tables
#   RESULTS.md  results summary with every number read from result files
#
# Run after eval_places365, tune_casv, run_casv_experiment and the two figure
# scripts. Missing inputs are reported, never filled in.
#
# Usage:
#   py -m scripts.build_journal_results --run results/audiocaps_test100
# =============================================================================

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import date
from pathlib import Path

OUT = Path("journal_results")


def write_table(name, header, rows, caption):
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    with open(OUT / "tables" / f"{name}.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    def tex(cell):
        return str(cell).replace("%", r"\%").replace("_", r"\_").replace("±", r"$\pm$")
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{tex(caption)}}}", rf"\label{{tab:{name}}}",
             r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}", r"\toprule",
             " & ".join(tex(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(tex(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + \
        "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)


def ci(block):
    return f"{block['mean']:.3f} [{block['ci95'][0]:.3f}, {block['ci95'][1]:.3f}]"


def main():
    parser = argparse.ArgumentParser(description="Assemble journal_results/ from experiment outputs.")
    parser.add_argument("--run", default="results/audiocaps_test100")
    parser.add_argument("--places", default="results/places365_eval.json")
    parser.add_argument("--tuning", default="results/tuning/tuning_results.json")
    args = parser.parse_args()
    run_dir = Path(args.run)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    figures = [
        (Path(args.places).with_name("fig_places365_recognition.png"), "fig1_scene_recognition.png"),
        (Path(args.tuning).with_name("fig_prompt_selection.png"), "fig2_prompt_selection.png"),
        (run_dir / "fig_clap_comparison.png", "fig3_generated_audio_clap.png"),
        (run_dir / "fig_verification_stats.png", "fig4_verification_statistics.png"),
    ] + [(p, f"fig5{chr(ord('a') + i)}_qualitative_examples.png")
         for i, p in enumerate(sorted(run_dir.glob("casv_comparison_*.png")))]
    missing = []
    for src, dest in figures:
        if src.exists():
            shutil.copy(src, OUT / "figures" / dest)
        else:
            missing.append(str(src))

    sections = []

    if Path(args.places).exists():
        p = json.loads(Path(args.places).read_text())["summary"]
        sections.append("## Table 1. Scene recognition on Places365 val\n\n" + write_table(
            "table1_scene_recognition", ["Metric", "Value", "n"],
            [["Scene top-1 accuracy (365-way)", f"{p['scene_top1']:.1%}", p["n_images"]],
             ["Scene top-5 accuracy", f"{p['scene_top5']:.1%}", p["n_images"]],
             ["Indoor/outdoor accuracy", f"{p['env_accuracy']:.1%}", p["n_images"]],
             ["Indoor/outdoor accuracy, p >= 0.7", f"{p['env_accuracy_when_p_ge_0.7']:.1%}",
              round(p["env_coverage_p_ge_0.7"] * p["n_images"])]],
            f"CLIP ({p['clip_model']}) zero-shot recognition on a stratified sample of Places365 val "
            "(10 images per class)."))
    else:
        missing.append(args.places)

    if Path(args.tuning).exists():
        t = json.loads(Path(args.tuning).read_text())
        rows = []
        for split in ("val", "test"):
            for method in ("baseline", "default", "tuned", "oracle"):
                rows.append([split, method, ci(t[split][method]), t[split]["n"]])
        sections.append("## Table 2. Prompt selection: CLAP(prompt, ground-truth recording)\n\n" + write_table(
            "table2_prompt_selection", ["Split", "Method", "Mean [95% CI]", "n"], rows,
            "CLAP cosine between the selected prompt and the real AudioCaps recording. "
            "Tuned on val only; test evaluated once."))
        d = t["test"]["tuned_minus_baseline"]
        sections.append(f"Test, tuned − baseline: {ci(d)}; better on {d['better']}, worse on {d['worse']}, "
                        f"unchanged on {d['same']} items.\n\nTuned configuration: `{json.dumps(t['tuned_config'])}`")
    else:
        missing.append(args.tuning)

    results_path = run_dir / "results.json"
    stats_path = run_dir / "graph_stats.json"
    if results_path.exists() and stats_path.exists():
        res = json.loads(results_path.read_text())
        stats = json.loads(stats_path.read_text())
        rows = []
        for key, label in (("clap_text", "CLAP(audio, GT captions)"), ("clap_gt_audio", "CLAP(audio, GT recording)")):
            if key in stats.get("clap", {}):
                c = stats["clap"][key]
                p_value = "n/a" if c["wilcoxon_p_changed_items"] is None else f"{c['wilcoxon_p_changed_items']:.3g}"
                rows.append([label, f"{c['baseline_mean']:.3f}", f"{c['casv_mean']:.3f}",
                             f"{c['mean_diff']:+.3f} [{c['mean_diff_ci95'][0]:+.3f}, {c['mean_diff_ci95'][1]:+.3f}]",
                             f"{c['casv_better']}/{c['casv_worse']}", p_value, c["n"]])
        sections.append("## Table 3. Generated audio (AudioLDM-S), baseline vs CAS-V\n\n" + write_table(
            "table3_generated_audio", ["Metric", "Baseline", "CAS-V", "Δ [95% CI]", "better/worse",
                                       "Wilcoxon p", "n"], rows,
            f"Same generator ({res['models']['generator']}) and seed for both arms; "
            f"CAS-V changed the prompt on {res['summary']['prompt_changed']} of {res['summary']['n_items']} items."))
        v = stats["verification"]
        conflicts = Counter(v["conflicts"]).most_common(8)
        sections.append("## Table 4. Verification statistics\n\n" + write_table(
            "table4_verification", ["Quantity", "Count"],
            [[f"Candidates {k}", v["candidate_decisions"].get(k, 0)] for k in ("ACCEPT", "REPAIR", "REJECT")]
            + [[f"Images {k}", v["item_status"].get(k, 0)] for k in ("ACCEPT", "REPAIR", "UNCERTAIN")]
            + [[f"Conflict: {name}", n] for name, n in conflicts],
            "Decisions over all retrieved candidate captions and final per-image status."))
    else:
        missing += [str(p) for p in (results_path, stats_path) if not p.exists()]

    header = f"""# CAS-V results

Generated {date.today().isoformat()} by `scripts/build_journal_results.py`. Every number below is read
from result files in `results/`; nothing is entered by hand.

## Setup

| Component | Choice |
|---|---|
| Recognition | CLIP ViT-B/32 zero-shot: Places365 scenes (365), ImageNet objects (1000) |
| Sound candidates | AudioCaps **train** captions (test/val captions never enter the candidate pool) |
| Verification | visual + scene + object support; explicit conflicts: indoor/outdoor, absent sound source |
| Generator | AudioLDM-S (`cvssp/audioldm-s-full-v2`), same seed for both arms |
| Evaluation | AudioCaps test clips: video frame as input, real recording + 5 human captions as ground truth; CLAP (`laion/clap-htsat-unfused`) |
| Baseline | same generator prompted with the top-1 retrieved caption, unverified |
| Tuning | grid search on AudioCaps **val**; test scored once |

## Figures

| File | Content |
|---|---|
| `figures/fig1_scene_recognition.png` | Places365 accuracy and confidence calibration |
| `figures/fig2_prompt_selection.png` | Prompt selection on val/test with 95% CIs |
| `figures/fig3_generated_audio_clap.png` | Generated audio: baseline vs CAS-V, paired |
| `figures/fig4_verification_statistics.png` | Candidate decisions and conflict types |
| `figures/fig5*_qualitative_examples.png` | First 20 test items in sample order (not selected) |

LaTeX versions of all tables are in `tables/`.
"""
    limitations = """
## Limitations

- Sample sizes are small (100 test pairs); report confidence intervals with every mean.
- CLAP similarity is an automatic proxy; no listening test has been run.
- AudioCaps frames are single video frames; off-screen sounds (narration, music) in the ground truth
  cannot be recovered from the image by any method.
- Places365/ImageNet are used for zero-shot vocabularies and Places365 val for recognition accuracy only;
  they contain no audio.
"""
    if missing:
        limitations += "\n## Missing inputs at build time\n\n" + "\n".join(f"- `{m}`" for m in missing) + "\n"
    (OUT / "RESULTS.md").write_text(header + "\n" + "\n\n".join(sections) + "\n" + limitations, encoding="utf-8")
    print(f"[+] {OUT.resolve()}")
    for m in missing:
        print(f"    missing: {m}")


if __name__ == "__main__":
    main()
