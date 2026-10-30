"""Reading run outputs: item order for figures, statistics for graphs."""

import json
import os

import numpy as np

from scripts.figures import load_records
from scripts.plots import earlier_overlap, paired_stats


def write_item(run_dir, item_id, mtime):
    item_dir = run_dir / item_id
    item_dir.mkdir()
    (item_dir / "casv.wav").write_bytes(b"")
    record = item_dir / "record.json"
    record.write_text(json.dumps({"id": item_id}))
    os.utime(record, (mtime, mtime))


def test_finished_run_keeps_manifest_order_over_file_times(tmp_path):
    write_item(tmp_path, "first", mtime=2000)  # e.g. generated in this run
    write_item(tmp_path, "second", mtime=1000)  # e.g. copied with --reuse-from
    assert [r["id"] for r in load_records(tmp_path)] == ["second", "first"]

    (tmp_path / "results.json").write_text(json.dumps({"items": [{"id": "first"}, {"id": "second"}]}))
    assert [r["id"] for r in load_records(tmp_path)] == ["first", "second"]


def test_earlier_overlap_reports_shared_items_and_config_changes(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps({"items": [{"id": "a"}, {"id": "z"}], "verifier_config": {"x": 2}})
    )
    results = {"items": [{"id": "a"}, {"id": "b"}], "verifier_config": {"x": 1}}
    overlap = earlier_overlap(results, [tmp_path])[0]
    assert (overlap["n_items"], overlap["n_shared"], overlap["same_config"]) == (2, 1, False)


def test_paired_stats_counts_and_skips_unscored_items():
    pairs = [(0.1, 0.2), (0.3, 0.2), (0.4, 0.4), (None, None)]
    items = [{"metrics": {"m_baseline": b, "m_casv": c}} for b, c in pairs]
    out = paired_stats(items, "m")
    assert (out["n"], out["n_prompt_changed"], out["casv_better"], out["casv_worse"]) == (3, 2, 1, 1)
    assert np.isclose(out["mean_diff"], 0.0)
    assert out["wilcoxon_p_changed_items"] is None  # fewer than 6 changed items
