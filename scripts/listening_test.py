# =============================================================================
# Blind A/B listening test: does the generated sound match what is VISIBLE?
#
#   build    package the items where CAS-V changed the prompt into a folder
#            raters open offline (index.html + image + two clips per item).
#            A/B order is randomised per item with --seed; the key is written
#            to results/listening_test/key.json, outside the rater package.
#   analyze  combine rater CSVs with the key: CAS-V preference with a two-sided
#            sign test (ties excluded), "contains a sound not in the image"
#            rates per system, and inter-rater agreement (Fleiss' kappa).
#
# Usage:
#   py -m scripts.listening_test build --run results/audiocaps_test100
#   py -m scripts.listening_test analyze --ratings evaluation/ratings/*.csv
# =============================================================================

import argparse
import csv
import glob
import html
import json
import logging
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from casv.log import setup_logging
from casv.stats import wilson_ci

log = logging.getLogger("casv")

PACKAGE_DIR = Path("evaluation/listening_test")
KEY_PATH = Path("results/listening_test/key.json")
OUT_DIR = Path("results/listening_test")
CHOICES = ("A", "B", "same")

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image-sound listening test</title>
<style>
:root {{ --ink:#111; --muted:#555; --line:#ddd; --accent:#2a78d6; --bg:#fcfcfb; }}
body {{ font-family: system-ui, sans-serif; color: var(--ink); background: var(--bg); max-width: 980px; margin: 0 auto; padding: 16px; }}
h1 {{ font-size: 1.4rem; }} .muted {{ color: var(--muted); }}
.item {{ border: 1px solid var(--line); border-radius: 10px; padding: 14px; margin: 18px 0; background: #fff; }}
.row {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }}
.row img {{ width: 300px; max-width: 100%; border-radius: 6px; }}
.clips {{ display: grid; gap: 8px; }} audio {{ width: 280px; max-width: 100%; }}
fieldset {{ border: none; padding: 0; margin: 8px 0; }} label {{ margin-right: 14px; }}
.bar {{ position: sticky; top: 0; background: var(--bg); padding: 8px 0; border-bottom: 1px solid var(--line); z-index: 1; }}
button {{ background: var(--accent); color: #fff; border: 0; border-radius: 6px; padding: 8px 14px; font-size: 1rem; cursor: pointer; }}
</style></head><body>
<h1>Does the sound match the picture?</h1>
<p>For each picture, listen to both sounds (A and B). Judge only <b>how well each sound fits what is visible in the picture</b>,
not audio quality. Use headphones if you can. There are {n} items; it takes about {minutes} minutes.</p>
<div class="bar">Your name or initials: <input id="rater" size="12"> &nbsp;
<span id="progress" class="muted"></span> &nbsp; <button onclick="download()">Download my ratings (CSV)</button></div>
{items}
<p><button onclick="download()">Download my ratings (CSV)</button> and send the file back. Thank you.</p>
<script>
const ITEMS = {ids};
function answered() {{ return ITEMS.filter(id => document.querySelector(`input[name="match_${{id}}"]:checked`)).length; }}
function update() {{ document.getElementById("progress").textContent = `${{answered()}} / ${{ITEMS.length}} answered`; }}
document.addEventListener("change", update); update();
function download() {{
  const rater = document.getElementById("rater").value.trim();
  if (!rater) {{ alert("Please enter your name or initials first."); return; }}
  if (answered() < ITEMS.length && !confirm(`${{ITEMS.length - answered()}} items are unanswered. Download anyway?`)) return;
  const rows = [["rater", "item", "better_match", "A_has_unrelated_sound", "B_has_unrelated_sound"]];
  for (const id of ITEMS) {{
    const m = document.querySelector(`input[name="match_${{id}}"]:checked`);
    rows.push([rater, id, m ? m.value : "", document.getElementById(`a_${{id}}`).checked, document.getElementById(`b_${{id}}`).checked]);
  }}
  const blob = new Blob([rows.map(r => r.join(",")).join("\\n")], {{type: "text/csv"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = `ratings_${{rater.replace(/\\W+/g, "_")}}.csv`; link.click();
}}
</script></body></html>
"""

ITEM = """<div class="item"><div class="muted">Item {number} of {n}</div><div class="row">
<img src="stimuli/{id}.jpg" alt="picture for item {number}">
<div class="clips">
<div>Sound A<br><audio controls preload="none" src="stimuli/{id}_A.wav"></audio></div>
<div>Sound B<br><audio controls preload="none" src="stimuli/{id}_B.wav"></audio></div>
</div></div>
<fieldset><b>Which sound better matches what is visible?</b><br>
<label><input type="radio" name="match_{id}" value="A"> A</label>
<label><input type="radio" name="match_{id}" value="B"> B</label>
<label><input type="radio" name="match_{id}" value="same"> No difference</label></fieldset>
<fieldset><b>Contains a sound that clearly does not belong to this picture:</b><br>
<label><input type="checkbox" id="a_{id}"> A</label> <label><input type="checkbox" id="b_{id}"> B</label></fieldset>
</div>"""


def build(run_dir: Path, seed: int) -> None:
    """Write the blinded rater package and the separate answer key."""
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    items = [r for r in results["items"] if r["prompts"]["baseline"] != r["prompts"]["casv"]]
    rng = random.Random(seed)
    rng.shuffle(items)

    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    (PACKAGE_DIR / "stimuli").mkdir(parents=True)
    key, blocks = {}, []
    for number, record in enumerate(items, 1):
        item_id = f"item{number:03d}"
        casv_is_a = rng.random() < 0.5
        source = run_dir / record["id"]
        shutil.copy(next(source.glob("input.*")), PACKAGE_DIR / "stimuli" / f"{item_id}.jpg")
        shutil.copy(
            source / ("casv.wav" if casv_is_a else "baseline.wav"), PACKAGE_DIR / "stimuli" / f"{item_id}_A.wav"
        )
        shutil.copy(
            source / ("baseline.wav" if casv_is_a else "casv.wav"), PACKAGE_DIR / "stimuli" / f"{item_id}_B.wav"
        )
        key[item_id] = {"audiocaps_id": record["id"], "casv": "A" if casv_is_a else "B", "prompts": record["prompts"]}
        blocks.append(ITEM.format(number=number, n=len(items), id=html.escape(item_id)))

    page = PAGE.format(
        n=len(items), minutes=round(len(items) * 0.5), items="\n".join(blocks), ids=json.dumps(list(key))
    )
    (PACKAGE_DIR / "index.html").write_text(page, encoding="utf-8")
    (PACKAGE_DIR / "README.txt").write_text(
        "Open index.html in a web browser (Chrome, Edge or Firefox), answer every item, enter your name,\n"
        "then click 'Download my ratings (CSV)' and send the CSV file back.\n",
        encoding="utf-8",
    )
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEY_PATH.write_text(
        json.dumps({"seed": seed, "run": str(run_dir), "n_items": len(items), "items": key}, indent=2), encoding="utf-8"
    )
    log.info(f"rater package: {PACKAGE_DIR} ({len(items)} items); key kept separately: {KEY_PATH}")
    log.info(f"CAS-V is clip A on {sum(v['casv'] == 'A' for v in key.values())} of {len(items)} items")


def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' kappa for an items x categories count table (same number of raters per item)."""
    n_raters = table.sum(axis=1)[0]
    p_cat = table.sum(axis=0) / table.sum()
    p_item = ((table**2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar, p_exp = p_item.mean(), (p_cat**2).sum()
    return float((p_bar - p_exp) / (1 - p_exp)) if p_exp < 1 else float("nan")


def analyze(rating_files: list[str]) -> None:
    """Unblind the ratings with the key and report preference, unrelated-sound rates and agreement."""
    key = json.loads(KEY_PATH.read_text(encoding="utf-8"))["items"]
    ratings = []
    for path in rating_files:
        with open(path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["item"] in key and row["better_match"] in CHOICES:
                    casv = key[row["item"]]["casv"]
                    choice = row["better_match"]
                    ratings.append(
                        {
                            "rater": row["rater"],
                            "item": row["item"],
                            "preferred": "same" if choice == "same" else ("casv" if choice == casv else "baseline"),
                            "casv_unrelated": row[f"{casv}_has_unrelated_sound"] == "true",
                            "baseline_unrelated": row[f"{'B' if casv == 'A' else 'A'}_has_unrelated_sound"] == "true",
                        }
                    )
    if not ratings:
        raise SystemExit("No valid ratings found.")

    counts = Counter(r["preferred"] for r in ratings)
    decisive = counts["casv"] + counts["baseline"]
    test = binomtest(counts["casv"], decisive, 0.5) if decisive else None
    raters = sorted({r["rater"] for r in ratings})

    by_item = {}
    for r in ratings:
        by_item.setdefault(r["item"], Counter())[r["preferred"]] += 1
    complete = [c for c in by_item.values() if sum(c.values()) == len(raters)]
    kappa = (
        fleiss_kappa(np.array([[c["casv"], c["baseline"], c["same"]] for c in complete]))
        if len(raters) > 1 and complete
        else None
    )

    summary = {
        "raters": raters,
        "n_ratings": len(ratings),
        "n_items": len(by_item),
        "preferred": dict(counts),
        "casv_win_rate_excluding_ties": counts["casv"] / decisive if decisive else None,
        "casv_win_rate_ci95": list(wilson_ci(counts["casv"], decisive)) if decisive else None,
        "sign_test_p": test.pvalue if test else None,
        "unrelated_sound_rate": {
            "casv": float(np.mean([r["casv_unrelated"] for r in ratings])),
            "baseline": float(np.mean([r["baseline_unrelated"] for r in ratings])),
        },
        "fleiss_kappa": kappa,
        "per_rater": {name: dict(Counter(r["preferred"] for r in ratings if r["rater"] == name)) for name in raters},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(json.dumps(summary, indent=2))


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Blind A/B listening test for image-sound match.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="Write the rater package and the answer key.")
    b.add_argument("--run", default="results/audiocaps_test100")
    b.add_argument("--seed", type=int, default=7)
    a = sub.add_parser("analyze", help="Unblind and analyse rater CSVs.")
    a.add_argument("--ratings", nargs="+", required=True, help="Rater CSV files (globs allowed).")
    args = parser.parse_args()

    if args.command == "build":
        build(Path(args.run), args.seed)
    else:
        analyze([path for pattern in args.ratings for path in glob.glob(pattern)])


if __name__ == "__main__":
    main()
