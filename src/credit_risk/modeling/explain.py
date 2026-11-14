"""SHAP explanations: global summary/importance and a per-applicant waterfall."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from credit_risk.scoring.feature_info import GROUPS, info


def shap_values(booster: lgb.Booster, X: pd.DataFrame) -> shap.Explanation:
    explainer = shap.TreeExplainer(booster)
    explanation = explainer(X)
    return explanation


def _labelled(explanation: shap.Explanation) -> shap.Explanation:
    names = [info(n).label for n in explanation.feature_names]
    return shap.Explanation(
        values=explanation.values,
        base_values=explanation.base_values,
        data=explanation.data,
        feature_names=names,
    )


def summary_plots(explanation: shap.Explanation, out_dir: Path, max_display: int = 20) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labelled = _labelled(explanation)
    # Beeswarm needs numeric data for colouring; categorical columns are shown in grey.
    data = np.array(
        [
            [v if isinstance(v, int | float) and not isinstance(v, bool) else np.nan for v in row]
            for row in labelled.data
        ],
        dtype="float64",
    )
    numeric = shap.Explanation(
        labelled.values, labelled.base_values, data, feature_names=labelled.feature_names
    )

    plt.figure()
    shap.plots.beeswarm(numeric, max_display=max_display, show=False, plot_size=(10, 7.5))
    plt.title("SHAP summary (hold-out sample): red = high feature value")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=130)
    plt.close("all")

    plt.figure()
    shap.plots.bar(numeric, max_display=max_display, show=False)
    plt.title("Mean |SHAP| (log-odds), top features")
    plt.gcf().set_size_inches(10, 7)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance.png", dpi=130)
    plt.close("all")


def waterfall(explanation: shap.Explanation, index: int, path: Path, title: str) -> None:
    labelled = _labelled(explanation)
    row = labelled[index]
    plt.figure()
    shap.plots.waterfall(row, max_display=12, show=False)
    plt.title(title)
    plt.gcf().set_size_inches(10, 6.5)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close("all")


def importance_tables(explanation: shap.Explanation) -> tuple[pd.Series, pd.Series]:
    mean_abs = pd.Series(np.abs(explanation.values).mean(axis=0), index=explanation.feature_names)
    by_feature = mean_abs.sort_values(ascending=False)
    groups = pd.Series({n: info(n).group for n in explanation.feature_names})
    # Group importance = mean over applicants of |sum of the group's SHAP values|.
    vals = pd.DataFrame(explanation.values, columns=explanation.feature_names)
    grouped = vals.T.groupby(groups).sum().T.abs().mean()
    grouped.index = [GROUPS[g][0] for g in grouped.index]
    return by_feature, grouped.sort_values(ascending=False)
