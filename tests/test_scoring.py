"""Checks against the committed model and synthetic sample."""

import math

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import PROJECT_ROOT
from credit_risk.features.application import add_ratio_features
from credit_risk.scoring.frame import prepare_frame, records_to_frame
from credit_risk.scoring.model import ScoringModel

SAMPLE = PROJECT_ROOT / "data" / "sample" / "synthetic_applicants.csv"


@pytest.fixture(scope="module")
def model():
    return ScoringModel.load(PROJECT_ROOT / "models")


@pytest.fixture(scope="module")
def sample(model):
    df = pd.read_csv(SAMPLE)
    return [
        {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in row.items()
            if k in model.input_features
        }
        for row in df.to_dict(orient="records")
    ]


def test_meta_consistent(model):
    meta = model.meta
    assert model.booster.feature_name() == meta["features"]
    assert set(meta["input_features"]) | set(meta["derived_features"]) == set(meta["features"])
    assert not set(meta["excluded_features"]) & set(meta["features"])
    assert 0 < model.threshold < 1


def test_sample_is_synthetic_and_scoreable(model, sample):
    df = pd.read_csv(SAMPLE)
    assert df["SYNTHETIC_ID"].str.startswith("SYN-").all()
    assert set(df["SEGMENT"]) == {"thin_file", "no_bureau_has_prev", "has_bureau"}
    scores = model.score(sample)
    probs = np.array([s.probability for s in scores])
    assert ((probs > 0) & (probs < 1)).all()
    assert {s.decision for s in scores} == {"approve", "decline"}
    thin = [s.thin_file for s, seg in zip(scores, df["SEGMENT"], strict=True) if seg == "thin_file"]
    assert all(thin)


def test_contributions_add_up_to_the_prediction(model, sample):
    X = model.frame(sample[:50])
    contrib = model.contributions(X)
    logit = np.log(model.predict_proba(X) / (1 - model.predict_proba(X)))
    assert np.allclose(contrib.sum(axis=1), logit, atol=1e-6)


def test_lightgbm_pred_contrib_equals_shap_tree_explainer(model, sample):
    shap = pytest.importorskip("shap")
    X = model.frame(sample[:100])
    native = model.contributions(X)
    explanation = shap.TreeExplainer(model.booster)(X)
    assert np.allclose(native[:, :-1], explanation.values, atol=1e-6)
    assert np.allclose(native[:, -1], explanation.base_values, atol=1e-6)


def test_api_frame_matches_training_frame(model, sample):
    """records_to_frame (API path) == add_ratio_features + prepare_frame (training path)."""
    api = records_to_frame(sample, model.features, model.categories)
    raw = pd.DataFrame(sample)
    train = prepare_frame(add_ratio_features(raw), model.features, model.categories)
    pd.testing.assert_frame_equal(api, train)


def test_unknown_category_becomes_missing(model, sample):
    rec = dict(sample[0], ORGANIZATION_TYPE="Not a real employer type")
    X = model.frame([rec])
    assert pd.isna(X.loc[0, "ORGANIZATION_TYPE"])
    assert 0 < model.predict_proba(X)[0] < 1
