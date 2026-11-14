import numpy as np
import pytest

from credit_risk.modeling.compare import compare
from credit_risk.modeling.metrics import bootstrap_ci, ks_statistic, reliability_table, summarize
from credit_risk.modeling.threshold import CostAssumptions, choose_threshold, outcomes_at


def test_ks_matches_definition():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.3, 0.7, 0.9])
    # Manual: max over cut-offs of (share of defaulters above) - (share of goods above)
    cuts = np.unique(p)
    manual = max(((p[y == 1] >= c).mean() - (p[y == 0] >= c).mean()) for c in cuts)
    assert ks_statistic(y, p) == pytest.approx(manual)
    assert ks_statistic(y, y.astype(float)) == pytest.approx(1.0)


def test_summarize_and_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, 2000), 0, 1)
    s = summarize(y, p)
    assert s["n"] == 2000 and 0.5 < s["roc_auc"] <= 1
    ci = bootstrap_ci(y, p, n_boot=200, seed=1)
    for key in ("roc_auc", "pr_auc", "ks", "brier"):
        lo, hi = ci[key]
        assert lo <= s[key] <= hi
    # a segment with a single class has no ranking metrics
    assert "roc_auc" not in summarize(np.zeros(5), np.full(5, 0.1))


def test_reliability_table_is_calibrated_for_calibrated_scores():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.5, 200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    t = reliability_table(y, p, n_bins=10)
    assert len(t) == 10 and t["n"].sum() == p.size
    assert np.allclose(t["predicted"], t["observed"], atol=0.01)


def test_outcomes_hand_computed():
    y = np.array([1, 0, 0, 1])
    p = np.array([0.9, 0.1, 0.5, 0.2])
    amt = np.array([100.0, 200.0, 300.0, 400.0])
    costs = CostAssumptions(loss_given_default=0.5, margin_good_loan=0.1)
    o = outcomes_at(y, p, amt, 0.4, costs)  # approve rows 1 and 3
    assert o["approval_rate"] == 0.5
    assert o["default_rate_approved"] == 0.5  # row 3 defaults
    assert o["cost_default_losses"] == pytest.approx(0.5 * 400)
    assert o["cost_lost_margin"] == pytest.approx(0.1 * 300)  # good row 2 declined
    assert o["defaults_caught"] == 0.5 and o["good_rejected"] == 0.5
    assert o["cost_approve_all"] == pytest.approx(0.5 * 500)
    assert o["saving_vs_approve_all"] == pytest.approx(1 - 230 / 250)


def test_threshold_lands_near_breakeven_for_calibrated_scores():
    rng = np.random.default_rng(2)
    p = rng.beta(1, 9, 300_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    amt = np.ones_like(p)
    costs = CostAssumptions(0.45, 0.10)
    t, curve = choose_threshold(y, p, amt, costs)
    assert costs.breakeven_probability == pytest.approx(0.10 / 0.55)
    assert abs(t - costs.breakeven_probability) <= 0.02
    assert curve["cost_total"].min() == pytest.approx(
        curve.loc[curve["threshold"] == t, "cost_total"].iloc[0]
    )


def test_compare_metrics_tolerance():
    def m(auc):
        seg = {
            "lightgbm": {"roc_auc": auc, "pr_auc": 0.2, "ks": 0.4, "brier": 0.07},
            "baseline": {"roc_auc": 0.75, "pr_auc": 0.2, "ks": 0.4, "brier": 0.07},
            "decision": {"approval_rate": 0.8, "default_rate_approved": 0.05},
        }
        return {"threshold": 0.2, "holdout": {"overall": seg, "thin_file": seg}}

    assert compare(m(0.790), m(0.793), tol=0.005)[1]
    assert not compare(m(0.790), m(0.80), tol=0.005)[1]
