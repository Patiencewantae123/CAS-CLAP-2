"""Unblinding and statistics of the listening test, on synthetic ratings in a temp folder."""

import json

import numpy as np
import pytest

from scripts import listening_test as lt


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "KEY_PATH", tmp_path / "key.json")
    monkeypatch.setattr(lt, "OUT_DIR", tmp_path / "out")
    items = {"item001": {"casv": "A"}, "item002": {"casv": "B"}, "item003": {"casv": "A"}}
    (tmp_path / "key.json").write_text(json.dumps({"package": "p1", "items": items}))
    return tmp_path


def write_ratings(path, rater, rows, package="p1"):
    lines = ["package,rater,item,better_match,A_has_unrelated_sound,B_has_unrelated_sound"]
    lines += [f"{package},{rater},{item},{choice},{a},{b}" for item, choice, a, b in rows]
    path.write_text("\n".join(lines))


def summary_of(workspace):
    return json.loads((workspace / "out" / "summary.json").read_text())


def test_unblinding_maps_choices_to_systems(workspace):
    # rater picks the CAS-V clip on items 1 and 2 (A, then B) and says "same" on item 3
    write_ratings(
        workspace / "r1.csv",
        "r1",
        [("item001", "A", "false", "true"), ("item002", "B", "true", "false"), ("item003", "same", "false", "false")],
    )
    lt.analyze([str(workspace / "r1.csv")])
    summary = summary_of(workspace)
    assert summary["preferred"] == {"casv": 2, "same": 1}
    assert summary["casv_win_rate_excluding_ties"] == 1.0
    # item001: baseline is B -> flagged; item002: baseline is A -> flagged
    assert summary["unrelated_sound_rate"]["baseline"] == pytest.approx(2 / 3)
    assert summary["unrelated_sound_rate"]["casv"] == 0.0
    assert summary["unrelated_sound_reduction"]["mean"] == pytest.approx(2 / 3)


def test_each_item_counts_once_however_many_raters(workspace):
    rows = [("item001", "A", "false", "false"), ("item002", "B", "false", "false"), ("item003", "A", "false", "false")]
    files = []
    for rater in ("r1", "r2", "r3"):
        write_ratings(workspace / f"{rater}.csv", rater, rows)
        files.append(str(workspace / f"{rater}.csv"))
    lt.analyze(files)
    summary = summary_of(workspace)
    assert summary["preferred"] == {"casv": 9}
    assert summary["item_winners"] == {"casv": 3}
    assert summary["sign_test_p"] == pytest.approx(0.25)  # 3 of 3 items, not 9 of 9 ratings


def test_excel_resaved_csv_is_read_correctly(workspace):
    write_ratings(workspace / "r1.csv", "r1", [("item001", "A", "FALSE", "TRUE")])
    text = (workspace / "r1.csv").read_text()
    (workspace / "r1.csv").write_text(text, encoding="utf-8-sig")
    lt.analyze([str(workspace / "r1.csv")])
    assert summary_of(workspace)["unrelated_sound_rate"]["baseline"] == 1.0


def test_ratings_from_another_build_are_refused(workspace):
    write_ratings(workspace / "r1.csv", "r1", [("item001", "A", "false", "false")], package="old")
    with pytest.raises(SystemExit, match="different build"):
        lt.analyze([str(workspace / "r1.csv")])


def test_item_rated_twice_under_one_name_is_refused(workspace):
    write_ratings(workspace / "r1.csv", "r1", [("item001", "A", "false", "false")])
    with pytest.raises(SystemExit, match="rated twice"):
        lt.analyze([str(workspace / "r1.csv"), str(workspace / "r1.csv")])


def test_item_winner_is_the_majority_and_ties_are_kept_apart():
    assert lt.item_winner(lt.Counter(casv=2, baseline=1)) == "casv"
    assert lt.item_winner(lt.Counter(casv=1, baseline=1, same=1)) == "tied"
    assert lt.item_winner(lt.Counter(baseline=1, same=2)) == "baseline"


def test_fleiss_kappa_perfect_and_chance():
    perfect = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    assert lt.fleiss_kappa(perfect) == pytest.approx(1.0)
    split = np.array([[1, 1, 1], [1, 1, 1]])
    assert lt.fleiss_kappa(split) < 0
