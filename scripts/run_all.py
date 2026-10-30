# =============================================================================
# Reproduce every reported result in order, from data to journal_results/.
#
# Each stage is a separate script; this runner only sequences them, stops on
# the first failure, and can skip stages whose outputs already exist.
#
# Usage:
#   py -m scripts.run_all                 # everything (hours on CPU)
#   py -m scripts.run_all --from figures  # only re-render figures + journal
#   py -m scripts.run_all --list
# =============================================================================

import argparse
import logging
import subprocess
import sys
import time

from casv.log import setup_logging

log = logging.getLogger("casv")

TEST_MANIFEST = "data/audiocaps/pairs/test/manifest.json"
TEST400_MANIFEST = "data/audiocaps/pairs/test400/manifest.json"
VAL_MANIFEST = "data/audiocaps/pairs/val/manifest.json"
CATEGORY_MANIFEST = "data/audiocaps/pairs/categories/manifest.json"
SCREENING = "data/audiocaps/pairs/categories/screening/screening.json"

STAGES = [
    ("data", "download_data", ["--split", "test", "--n", "100"]),
    ("data", "download_data", ["--split", "test", "--n", "400", "--out", "data/audiocaps/pairs/test400"]),
    ("data", "download_data", ["--split", "val", "--n", "100"]),
    ("data", "select_examples", ["--screening", SCREENING]),
    ("places", "eval_scenes", ["--per-class", "10"]),
    ("train", "train", ["--val", VAL_MANIFEST, "--test", TEST400_MANIFEST, "--out", "results/tuning_test400"]),
    ("test", "test", ["--manifest", TEST_MANIFEST, "--run-name", "audiocaps_test100"]),
    (
        "test",
        "test",
        [
            "--manifest",
            TEST400_MANIFEST,
            "--run-name",
            "audiocaps_test400",
            "--reuse-from",
            "results/audiocaps_test100",
        ],
    ),
    ("test", "test", ["--manifest", CATEGORY_MANIFEST, "--run-name", "categories7"]),
    (
        "figures",
        "plots",
        [
            "--run",
            "results/audiocaps_test400",
            "--earlier-run",
            "results/audiocaps_test100",
            "results/categories7",
        ],
    ),
    ("figures", "figures", ["--run", "results/audiocaps_test400", "--count", "20"]),
    ("figures", "figures", ["--run", "results/audiocaps_test400", "--count", "140", "--folder", "comparisons_140"]),
    ("figures", "figures", ["--run", "results/categories7", "--count", "7"]),
    ("journal", "report", []),
]
GROUPS = list(dict.fromkeys(group for group, _, _ in STAGES))


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Run the CAS-V pipeline end to end.")
    parser.add_argument("--from", dest="start", choices=GROUPS, default=GROUPS[0], help="First stage group to run.")
    parser.add_argument("--only", choices=GROUPS, default=None, help="Run a single stage group.")
    parser.add_argument("--list", action="store_true", help="Print the stages and exit.")
    args = parser.parse_args()

    if args.only:
        selected = [s for s in STAGES if s[0] == args.only]
    else:
        selected = STAGES[next(i for i, s in enumerate(STAGES) if s[0] == args.start) :]

    for group, script, script_args in selected:
        command = [sys.executable, "-m", f"scripts.{script}", *script_args]
        if args.list:
            print(f"{group:11} {' '.join(command[1:])}")
            continue
        log.info(f"[{group}] {script} {' '.join(script_args)}")
        start = time.time()
        if subprocess.run(command).returncode != 0:
            log.error(f"[{group}] {script} failed; stopping")
            sys.exit(1)
        log.info(f"[{group}] {script} done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
