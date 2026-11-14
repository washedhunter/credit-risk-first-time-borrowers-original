"""Evaluation plots (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from credit_risk.modeling.metrics import reliability_table

COLORS = {
    "overall": "#1f4e79",
    "thin_file": "#c0504d",
    "no_bureau": "#e8a33d",
    "baseline": "#7f7f7f",
}
LABELS = {
    "overall": "LightGBM, all applicants",
    "thin_file": "LightGBM, thin-file",
    "no_bureau": "LightGBM, no bureau record",
    "baseline": "Logistic baseline, all applicants",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def roc_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    """curves: name -> (y, p) on the hold-out."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        style = "--" if name == "baseline" else "-"
        ax1.plot(fpr, tpr, style, color=COLORS[name], label=f"{LABELS[name]} (n={len(y):,})")
        ax2.plot(rec, prec, style, color=COLORS[name], label=LABELS[name])
        ax2.axhline(np.mean(y), color=COLORS[name], lw=0.6, alpha=0.4)
    ax1.plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (hold-out)")
    ax2.set(
        xlabel="Recall (defaults caught)", ylabel="Precision", title="Precision-recall (hold-out)"
    )
    ax1.legend(fontsize=8, loc="lower right")
    ax2.legend(fontsize=8, loc="upper right")
    _save(fig, path)


def calibration(
    holdout: dict[str, tuple[np.ndarray, np.ndarray, int]],
    oof_strategies: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, (y, p, bins) in holdout.items():
        t = reliability_table(y, p, n_bins=bins)
        ax1.plot(
            t["predicted"],
            t["observed"],
            "o-",
            color=COLORS[name],
            label=f"{LABELS[name]} ({bins} bins)",
        )
    for name, (y, p) in oof_strategies.items():
        t = reliability_table(y, p, n_bins=10)
        ax2.plot(t["predicted"], t["observed"], "o-", label=f"scale_pos_weight: {name}")
    for ax in (ax1, ax2):
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], ":", color="grey", lw=0.8)
        ax.set(xlabel="Mean predicted default probability", ylabel="Observed default rate")
        ax.legend(fontsize=8)
    ax1.set_title("Reliability, chosen model (hold-out)")
    ax2.set_title("Reliability by imbalance strategy (out-of-fold, train)")
    _save(fig, path)


def cost_vs_threshold(
    curves: dict[str, pd.DataFrame], chosen: float, path: Path, breakeven: float | None = None
) -> None:
    """curves: label -> cost_curve() frame. Plots cost per application and approval rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for label, c in curves.items():
        ax1.plot(c["threshold"], c["cost_per_application"], label=label)
        ax2.plot(c["threshold"], c["approval_rate"], label=label)
    for ax in (ax1, ax2):
        ax.axvline(chosen, color="black", lw=0.9, ls="--")
        ax.set_xlabel("Threshold (decline when P(default) ≥ threshold)")
        ax.set_xlim(0, 0.6)
    if breakeven is not None:
        ax1.axvline(breakeven, color="grey", lw=0.8, ls=":")
    ax1.set(
        ylabel="Expected cost per application",
        title=f"Expected cost vs threshold (chosen {chosen:.3f})",
    )
    ax2.set(ylabel="Approval rate", title="Approval rate vs threshold")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    _save(fig, path)


def group_importance(groups: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    groups = groups.sort_values()
    ax.barh(groups.index, groups.to_numpy(), color=COLORS["overall"])
    ax.set(
        xlabel="Mean |SHAP| (log-odds), summed within reason group",
        title="What drives the score (hold-out sample)",
    )
    _save(fig, path)
