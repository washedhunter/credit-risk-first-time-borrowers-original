"""Synthetic sample generator and an end-to-end training run on fake data."""

import json

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import Settings
from credit_risk.features.application import CATEGORICAL
from credit_risk.features.build import input_feature_columns
from credit_risk.modeling.compare import compare
from credit_risk.synth import fit_sample, make_synthetic_sample


def test_fit_sample_preserves_marginals_and_missingness():
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "a": x,
            "b": np.where(rng.uniform(size=n) < 0.3, np.nan, 2 * x + rng.normal(scale=0.1, size=n)),
            "c": rng.choice(["u", "v", "w"], size=n, p=[0.6, 0.3, 0.1]),
            "k": rng.integers(0, 5, size=n).astype(float),
        }
    )
    out = fit_sample(df, ["a", "b", "c", "k"], ["c"], n=4000, seed=1)
    assert abs(out["b"].isna().mean() - 0.3) < 0.03
    assert set(out["c"].dropna()) <= {"u", "v", "w"}
    assert abs((out["c"] == "u").mean() - 0.6) < 0.04
    assert np.allclose(out["k"].dropna() % 1, 0)  # integer columns stay integer
    ok = out[["a", "b"]].dropna().astype(float)
    assert np.corrcoef(ok["a"], ok["b"])[0, 1] > 0.9  # dependence kept
    assert abs(out["a"].median() - df["a"].median()) < 0.1


def test_synthetic_sample_never_copies_real_rows(features):
    cols = input_feature_columns(features)
    sample = make_synthetic_sample(features, cols, CATEGORICAL, seed=3)
    assert len(sample) == 300 and sample["SYNTHETIC_ID"].is_unique
    numeric = [c for c in cols if c not in CATEGORICAL and features[c].nunique() > 50]
    real = {tuple(np.round(r, 6)) for r in features[numeric].fillna(-999).to_numpy()}
    synth = [tuple(np.round(r, 6)) for r in sample[numeric].fillna(-999).astype(float).to_numpy()]
    assert not real & set(synth)
    thin = sample[sample["SEGMENT"] == "thin_file"]
    assert (thin["BUR_COUNT"] == 0).all() and (thin["PREV_COUNT"] == 0).all()
    assert (sample["BUR_ACTIVE_COUNT"].fillna(0) <= sample["BUR_COUNT"]).all()


@pytest.mark.slow
def test_end_to_end_training_on_fake_data(features, tmp_path, monkeypatch):
    from credit_risk.modeling.train import train
    from credit_risk.scoring.model import ScoringModel

    data_dir = tmp_path / "data"
    (data_dir / "processed").mkdir(parents=True)
    features.to_parquet(data_dir / "processed" / "features.parquet", index=False)
    settings = Settings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        img_dir=tmp_path / "img",
        n_folds=3,
    )
    results = train(settings, log=lambda _: None)

    assert results["holdout"]["overall"]["lightgbm"]["roc_auc"] > 0.6  # fake data has real signal
    assert results["imbalance"]["chosen"] in {"none", "sqrt", "balanced"}
    for name in (
        "roc_pr.png",
        "calibration.png",
        "cost_threshold.png",
        "shap_summary.png",
        "shap_waterfall.png",
    ):
        assert (tmp_path / "img" / name).stat().st_size > 10_000
    metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert compare(metrics, metrics, tol=0.0)[1]
    assert "Thin-file" in (tmp_path / "reports" / "metrics.md").read_text()
    assert results["shap"]["max_abs_diff_shap_vs_lightgbm_pred_contrib"] < 1e-6

    model = ScoringModel.load(tmp_path / "models")
    sample = pd.read_csv(data_dir / "sample" / "synthetic_applicants.csv")
    records = (
        sample[model.input_features]
        .astype(object)
        .where(sample[model.input_features].notna(), None)
    )
    scores = model.score(records.to_dict(orient="records"))
    assert len(scores) == len(sample)
