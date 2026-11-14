"""Ranking, calibration and separation metrics, with bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max distance between the score CDFs of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "roc_auc": lambda y, p: float(roc_auc_score(y, p)),
    "pr_auc": lambda y, p: float(average_precision_score(y, p)),
    "ks": ks_statistic,
    "brier": lambda y, p: float(brier_score_loss(y, p)),
}


def summarize(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    p = np.asarray(p)
    out: dict[str, float] = {
        "n": len(y),
        "defaults": int(y.sum()),
        "default_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),
    }
    if 0 < y.sum() < len(y):
        out.update({name: fn(y, p) for name, fn in METRICS.items()})
    return out


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, list[float]]:
    """Percentile bootstrap CI for each metric (resampling applicants with replacement)."""
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in METRICS}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        pb = p[idx]
        for name, fn in METRICS.items():
            draws[name].append(fn(yb, pb))
    return {
        name: [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))]
        for name, v in draws.items()
        if v
    }


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predicted probability: mean predicted vs observed default rate."""
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), q=n_bins, labels=False)
    table = frame.groupby("bin").agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")
    )
    return table.reset_index(drop=True)
