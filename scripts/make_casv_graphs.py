# =============================================================================
# Result graphs for CAS-V. Every value is read from experiment outputs:
#   results/places365/eval.json          (eval_places365.py)
#   results/<run>/results.json + record.json per item (run_casv_experiment.py)
#
# Figures (300 dpi PNG, written next to the inputs):
#   fig_places365_recognition.png  scene / environment accuracy + calibration
#   fig_clap_comparison.png        baseline vs CAS-V, paired, with 95% CIs
#   fig_verification_stats.png     candidate decisions and conflict types
# Statistics used in the figures are saved to graph_stats.json.
#
# Usage:
#   py -m scripts.make_casv_graphs --run results/audiocaps_test100
# =============================================================================

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from casv.log import setup_logging
from casv.stats import bootstrap_ci, wilson_ci

log = logging.getLogger("casv")

# Reference palette (dataviz skill): categorical slots in fixed order + ink
CASV = "#2a78d6"  # slot 1
BASELINE = "#eb6834"  # slot 2
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def label_bars(ax, bars, fmt, tops=None):
    """Annotate bars with their values, above the error bar when given."""
    for i, bar in enumerate(bars):
        top = tops[i] if tops is not None else bar.get_height()
        ax.annotate(
            fmt(bar.get_height()),
            (bar.get_x() + bar.get_width() / 2, top),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK,
        )


# -----------------------------------------------------------------------------
# Figure 1: Places365 recognition
# -----------------------------------------------------------------------------


def places_figure(places_path, stats):
    """Figure 1: Places365 accuracy and confidence calibration."""
    data = json.loads(Path(places_path).read_text())
    rows, summary = data["images"], data["summary"]
    n = len(rows)

    confident = [r for r in rows if max(r["p_indoor"], 1 - r["p_indoor"]) >= 0.7]
    metrics = [
        ("Scene\ntop-1", sum(r["top1"] for r in rows), n),
        ("Scene\ntop-5", sum(r["top5"] for r in rows), n),
        ("Indoor /\noutdoor", sum(r["env_correct"] for r in rows), n),
        (
            f"Indoor / outdoor\np ≥ 0.7 ({len(confident) / n:.0%})",
            sum(r["env_correct"] for r in confident),
            len(confident),
        ),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
    values = [k / m for _, k, m in metrics]
    cis = [wilson_ci(k, m) for _, k, m in metrics]
    bars = ax1.bar(
        range(len(metrics)),
        values,
        width=0.6,
        color=CASV,
        yerr=np.array([[v - lo, hi - v] for v, (lo, hi) in zip(values, cis, strict=True)]).T,
        error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 3},
    )
    label_bars(ax1, bars, lambda v: f"{v:.1%}", tops=[hi for _, hi in cis])
    ax1.set_xticks(range(len(metrics)), [m[0] for m in metrics], fontsize=8)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"(a) CLIP zero-shot on Places365 val (n = {n})", fontsize=9, loc="left", color=INK)
    ax1.grid(axis="x", visible=False)

    edges = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    conf = np.array([r["confidence"] for r in rows])
    correct = np.array([r["top1"] for r in rows], dtype=float)
    centres, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() >= 10:
            centres.append(conf[mask].mean())
            accs.append(correct[mask].mean())
            counts.append(int(mask.sum()))
    ax2.plot([0, 1], [0, 1], linestyle="--", color=INK_2, linewidth=0.8, label="Perfect calibration")
    ax2.plot(centres, accs, color=CASV, linewidth=2, marker="o", markersize=5, label="CAS-V scene recognizer")
    for x, y, c in zip(centres, accs, counts, strict=True):
        ax2.annotate(f"n={c}", (x, y), xytext=(-6, 8), textcoords="offset points", ha="right", fontsize=7, color=INK_2)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Top-1 scene confidence")
    ax2.set_ylabel("Top-1 accuracy")
    ax2.set_title("(b) Confidence calibration (scene top-1)", fontsize=9, loc="left", color=INK)
    ax2.legend(fontsize=7.5, loc="upper left")

    out = Path(places_path).with_name("fig_places365_recognition.png")
    fig.savefig(out)
    plt.close(fig)
    stats["places365"] = {
        **summary,
        "calibration_bins": list(zip(map(float, centres), map(float, accs), counts, strict=True)),
    }
    return out


# -----------------------------------------------------------------------------
# Figure 2: CLAP comparison
# -----------------------------------------------------------------------------


def clap_figure(run_dir, results, stats):
    """Figure 3: generated audio, baseline vs CAS-V, paired with 95% CIs."""
    items = results["items"]
    panels = [("clap_text", "CLAP: audio vs GT captions"), ("clap_gt_audio", "CLAP: audio vs GT recording")]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    stats["clap"] = {}
    for ax, (key, title) in zip(axes[:2], panels, strict=True):
        pairs = np.array(
            [
                (r["metrics"][f"{key}_baseline"], r["metrics"][f"{key}_casv"])
                for r in items
                if r["metrics"].get(f"{key}_baseline") is not None
            ]
        )
        base, casv = pairs[:, 0], pairs[:, 1]
        diff = casv - base
        changed = diff != 0
        p_value = float(wilcoxon(casv[changed], base[changed]).pvalue) if changed.sum() >= 6 else None
        entry = {
            "n": int(len(pairs)),
            "n_prompt_changed": int(changed.sum()),
            "baseline_mean": float(base.mean()),
            "baseline_ci95": bootstrap_ci(base),
            "casv_mean": float(casv.mean()),
            "casv_ci95": bootstrap_ci(casv),
            "mean_diff": float(diff.mean()),
            "mean_diff_ci95": bootstrap_ci(diff),
            "casv_better": int((diff > 0).sum()),
            "casv_worse": int((diff < 0).sum()),
            "wilcoxon_p_changed_items": p_value,
        }
        stats["clap"][key] = entry

        means = [entry["baseline_mean"], entry["casv_mean"]]
        cis = [entry["baseline_ci95"], entry["casv_ci95"]]
        bars = ax.bar(
            [0, 1],
            means,
            width=0.55,
            color=[BASELINE, CASV],
            yerr=np.array([[m - lo, hi - m] for m, (lo, hi) in zip(means, cis, strict=True)]).T,
            error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 3},
        )
        label_bars(ax, bars, lambda v: f"{v:.3f}", tops=[hi for _, hi in cis])
        ax.set_xticks([0, 1], ["Baseline\n(unverified)", "CAS-V\n(verified)"])
        ax.set_ylabel("Mean cosine similarity")
        ax.set_ylim(0, max(hi for _, hi in cis) * 1.25)
        p_text = f"p = {p_value:.3g}" if p_value is not None else "p: too few changed items"
        ax.set_title(
            f"{title}\nn = {entry['n']}, Δ = {entry['mean_diff']:+.3f}, Wilcoxon {p_text}",
            fontsize=8.5,
            loc="left",
            color=INK,
        )
        ax.grid(axis="x", visible=False)

    key = "clap_gt_audio"
    pairs = np.array(
        [
            (r["metrics"][f"{key}_baseline"], r["metrics"][f"{key}_casv"])
            for r in items
            if r["prompts"]["baseline"] != r["prompts"]["casv"] and r["metrics"].get(f"{key}_baseline") is not None
        ]
    )
    ax = axes[2]
    if len(pairs):
        better = pairs[:, 1] > pairs[:, 0]
        lo, hi = pairs.min() - 0.03, pairs.max() + 0.03
        ax.plot([lo, hi], [lo, hi], linestyle="--", color=INK_2, linewidth=0.8)
        ax.scatter(
            *pairs[better].T, s=28, color=CASV, edgecolor="white", linewidth=1, label=f"CAS-V better ({better.sum()})"
        )
        ax.scatter(
            *pairs[~better].T,
            s=28,
            color=BASELINE,
            edgecolor="white",
            linewidth=1,
            label=f"Baseline better ({(~better).sum()})",
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.legend(fontsize=7.5, loc="upper left")
    ax.set_xlabel("Baseline CLAP (vs GT recording)")
    ax.set_ylabel("CAS-V CLAP (vs GT recording)")
    ax.set_title(f"Per item, where CAS-V changed the prompt (n = {len(pairs)})", fontsize=8.5, loc="left", color=INK)

    fig.tight_layout()
    out = run_dir / "fig_clap_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# Figure 3: verification statistics
# -----------------------------------------------------------------------------


def verification_figure(run_dir, results, stats):
    """Figure 4: candidate decisions, conflict types and final status."""
    decisions, conflicts, status = Counter(), Counter(), Counter()
    for item in results["items"]:
        record = json.loads((run_dir / item["id"] / "record.json").read_text())
        status[record["status"]] += 1
        for verdict in record["verdicts"]:
            decisions[verdict["decision"]] += 1
            for conflict in verdict["conflicts"]:
                if conflict.startswith("names '"):
                    conflicts["absent source: " + conflict.split("'")[1]] += 1
                else:
                    conflicts["environment: " + conflict.split(",")[0]] += 1
    stats["verification"] = {
        "candidate_decisions": dict(decisions),
        "conflicts": dict(conflicts),
        "item_status": dict(status),
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [0.8, 1.6, 0.8]})
    order = ["ACCEPT", "REPAIR", "REJECT"]
    total = sum(decisions.values())
    bars = ax1.bar(range(3), [decisions[d] for d in order], width=0.6, color=CASV)
    for bar, d in zip(bars, order, strict=True):
        ax1.annotate(
            f"{decisions[d]}\n({decisions[d] / total:.0%})",
            (bar.get_x() + 0.3, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK,
        )
    ax1.set_xticks(range(3), order)
    ax1.set_ylabel("Candidate captions")
    ax1.set_ylim(0, max(decisions.values()) * 1.25)
    ax1.set_title(f"(a) Candidate decisions (n = {total})", fontsize=9, loc="left", color=INK)
    ax1.grid(axis="x", visible=False)

    top = conflicts.most_common(12)[::-1]
    ax2.barh(range(len(top)), [c for _, c in top], height=0.6, color=CASV)
    ax2.set_yticks(range(len(top)), [name for name, _ in top], fontsize=7.5)
    for i, (_, count) in enumerate(top):
        ax2.annotate(
            str(count), (count, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK
        )
    ax2.set_xlabel("Conflicts detected")
    ax2.set_title("(b) Most frequent explicit conflicts", fontsize=9, loc="left", color=INK)
    ax2.grid(axis="y", visible=False)

    s_order = ["ACCEPT", "REPAIR", "UNCERTAIN"]
    n_items = sum(status.values())
    bars = ax3.bar(range(3), [status[s] for s in s_order], width=0.6, color=CASV)
    for bar, s in zip(bars, s_order, strict=True):
        ax3.annotate(
            str(status[s]),
            (bar.get_x() + 0.3, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK,
        )
    ax3.set_xticks(range(3), s_order, fontsize=8)
    ax3.set_ylabel("Images")
    ax3.set_ylim(0, max(status.values()) * 1.25)
    ax3.set_title(f"(c) Final status per image (n = {n_items})", fontsize=9, loc="left", color=INK)
    ax3.grid(axis="x", visible=False)

    fig.tight_layout()
    out = run_dir / "fig_verification_stats.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# Figure 4: prompt selection (tuned on val, evaluated once on test)
# -----------------------------------------------------------------------------


def tuning_figure(tuning_path, stats):
    """Figure 2: prompt selection on val and test."""
    data = json.loads(Path(tuning_path).read_text())
    methods = [
        ("baseline", "Baseline\n(top-1, unverified)"),
        ("default", "CAS-V\n(default)"),
        ("tuned", "CAS-V\n(tuned on val)"),
        ("oracle", "Oracle\n(best of k)"),
    ]
    colors = [BASELINE, CASV, CASV, INK_2]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, split in zip(axes, ("val", "test"), strict=True):
        block = data[split]
        means = [block[m]["mean"] for m, _ in methods]
        cis = [block[m]["ci95"] for m, _ in methods]
        bars = ax.bar(
            range(4),
            means,
            width=0.6,
            color=colors,
            yerr=np.array([[m - lo, hi - m] for m, (lo, hi) in zip(means, cis, strict=True)]).T,
            error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 3},
        )
        bars[2].set_hatch("////")
        bars[2].set_edgecolor("white")
        label_bars(ax, bars, lambda v: f"{v:.3f}", tops=[hi for _, hi in cis])
        ax.set_xticks(range(4), [label for _, label in methods], fontsize=7.5)
        d, t = block["default_minus_baseline"], block["tuned_minus_baseline"]
        ax.set_title(
            f"{split.upper()} (n = {block['n']})\n"
            f"default − baseline {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]\n"
            f"tuned − baseline {t['mean']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]",
            fontsize=8,
            loc="left",
            color=INK,
        )
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("CLAP(prompt, GT recording)")
    axes[0].set_ylim(0, max(block["oracle"]["ci95"][1] for block in (data["val"], data["test"])) * 1.2)

    fig.tight_layout()
    out = Path(tuning_path).with_name("fig_prompt_selection.png")
    fig.savefig(out)
    plt.close(fig)
    stats["tuning"] = {k: data[k] for k in ("tuned_config", "val", "test")}
    return out


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Render CAS-V result graphs from experiment outputs.")
    parser.add_argument("--run", default=None, help="Run directory from run_casv_experiment.py")
    parser.add_argument("--places", default="results/places365/eval.json")
    parser.add_argument("--tuning", default="results/tuning/tuning_results.json")
    args = parser.parse_args()

    stats, outputs = {}, []
    if Path(args.places).exists():
        outputs.append(places_figure(args.places, stats))
    if Path(args.tuning).exists():
        outputs.append(tuning_figure(args.tuning, stats))
    if args.run and not (Path(args.run) / "results.json").exists():
        log.warning(f"{args.run}/results.json not found (run unfinished?); skipping figures 3 and 4")
    elif args.run:
        run_dir = Path(args.run)
        results = json.loads((run_dir / "results.json").read_text())
        if results["models"]["clap"]:
            outputs.append(clap_figure(run_dir, results, stats))
        outputs.append(verification_figure(run_dir, results, stats))
        (run_dir / "graph_stats.json").write_text(json.dumps(stats, indent=2))
    for out in outputs:
        log.info(f"saved {out}")


if __name__ == "__main__":
    main()
