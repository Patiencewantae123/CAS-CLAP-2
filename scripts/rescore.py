# =============================================================================
# RESCORE: recompute the CLAP metrics of a finished run from its saved audio,
# without generating anything. Updates results.json and every record.json.
#
# Needed when the scorer changes. Runs scored before CLAPScorer fixed its crop
# of clips over 10 s depend on an unseeded random offset and cannot be repeated.
# A run made from scratch needs no rescoring.
#
# Usage:
#   py -m scripts.rescore --run results/audiocaps_test400
# =============================================================================

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from casv.audio import CLAPScorer
from casv.log import setup_logging
from scripts.test import score_item, summarise

log = logging.getLogger("casv")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Recompute CLAP metrics of a finished run from its saved audio.")
    parser.add_argument("--run", required=True, help="Run directory produced by test.py")
    args = parser.parse_args()

    run_dir = Path(args.run)
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text())
    scorer = CLAPScorer()

    for index, item in enumerate(results["items"], 1):
        item_dir = run_dir / item["id"]
        item["metrics"] = score_item(scorer, item_dir, item)
        record_path = item_dir / "record.json"
        record = json.loads(record_path.read_text())
        record["metrics"] = item["metrics"]
        record_path.write_text(json.dumps(record, indent=2))
        log.info(f"[{index}/{len(results['items'])}] {item['id']}")

    results["summary"] = summarise(results["items"])
    results["rescored"] = datetime.now().isoformat()
    results_path.write_text(json.dumps(results, indent=2))
    log.info(f"[+] Rescored {run_dir.resolve()}")


if __name__ == "__main__":
    main()
