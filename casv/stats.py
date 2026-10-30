"""Statistics used for every reported number: confidence intervals and paired comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

N_BOOTSTRAP = 10_000


def bootstrap_ci(values: Sequence[float], n_boot: int = N_BOOTSTRAP, seed: int = 0) -> list[float]:
    """Percentile bootstrap 95% confidence interval of the mean."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def paired_difference(treatment: Sequence[float], control: Sequence[float]) -> dict:
    """Mean paired difference (treatment - control) with bootstrap CI and win/loss counts."""
    diff = np.asarray(treatment, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    return {
        "mean": float(diff.mean()),
        "ci95": bootstrap_ci(diff),
        "better": int((diff > 0).sum()),
        "worse": int((diff < 0).sum()),
        "same": int((diff == 0).sum()),
    }
