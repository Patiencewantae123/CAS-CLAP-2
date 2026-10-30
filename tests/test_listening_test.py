"""Unblinding and statistics of the listening test, on synthetic ratings in a temp folder."""

import json

import numpy as np
import pytest

from scripts import listening_test as lt


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "KEY_PATH", tmp_path / "key.json")
    monkeypatch.setattr(lt, "OUT_DIR", tmp_path / "out")
    key = {"items": {"item001": {"casv": "A"}, "item002": {"casv": "B"}, "item003": {"casv": "A"}}}
    (tmp_path / "key.json").write_text(json.dumps(key))
    return tmp_path


def write_ratings(path, rater, rows):
    lines = ["rater,item,better_match,A_has_unrelated_sound,B_has_unrelated_sound"]
    lines += [f"{rater},{item},{choice},{a},{b}" for item, choice, a, b in rows]
    path.write_text("\n".join(lines))


def test_unblinding_maps_choices_to_systems(workspace):
    # rater picks the CAS-V clip on items 1 and 2 (A, then B) and says "same" on item 3
    write_ratings(
        workspace / "r1.csv",
        "r1",
        [("item001", "A", "false", "true"), ("item002", "B", "true", "false"), ("item003", "same", "false", "false")],
    )
    lt.analyze([str(workspace / "r1.csv")])
    summary = json.loads((workspace / "out" / "summary.json").read_text())
    assert summary["preferred"] == {"casv": 2, "same": 1}
    assert summary["casv_win_rate_excluding_ties"] == 1.0
    # item001: baseline is B -> flagged; item002: baseline is A -> flagged
    assert summary["unrelated_sound_rate"]["baseline"] == pytest.approx(2 / 3)
    assert summary["unrelated_sound_rate"]["casv"] == 0.0


def test_fleiss_kappa_perfect_and_chance():
    perfect = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    assert lt.fleiss_kappa(perfect) == pytest.approx(1.0)
    split = np.array([[1, 1, 1], [1, 1, 1]])
    assert lt.fleiss_kappa(split) < 0
