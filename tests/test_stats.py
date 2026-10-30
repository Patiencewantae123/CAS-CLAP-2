"""Confidence intervals and paired comparisons."""

import numpy as np

from casv.stats import bootstrap_ci, paired_difference, wilson_ci


def test_bootstrap_ci_is_reproducible_and_brackets_mean():
    values = np.random.default_rng(0).normal(0.3, 0.1, 100)
    lo, hi = bootstrap_ci(values)
    assert bootstrap_ci(values) == [lo, hi]
    assert lo < values.mean() < hi


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(95, 100)
    assert 0 <= lo < 0.95 < hi <= 1
    assert wilson_ci(0, 10)[0] == 0 or wilson_ci(0, 10)[0] < 1e-12


def test_paired_difference_counts():
    out = paired_difference([1.0, 2.0, 3.0], [1.0, 1.0, 4.0])
    assert (out["better"], out["worse"], out["same"]) == (1, 1, 1)
    assert np.isclose(out["mean"], 0.0)
