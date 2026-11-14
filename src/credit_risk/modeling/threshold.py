"""Expected-cost approval threshold.

Decision rule: approve when the predicted default probability is below the threshold.
Costs are proportional to the loan amount:
  approved defaulter    -> loss_given_default * AMT_CREDIT
  rejected good customer -> margin_good_loan  * AMT_CREDIT (profit we never earn)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

DEFAULT_GRID = np.round(np.arange(0.01, 0.951, 0.005), 3)


@dataclass(frozen=True)
class CostAssumptions:
    loss_given_default: float = 0.45
    margin_good_loan: float = 0.10

    @property
    def breakeven_probability(self) -> float:
        """For a calibrated model, approving is worth it while p < margin / (margin + LGD)."""
        return self.margin_good_loan / (self.margin_good_loan + self.loss_given_default)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def outcomes_at(
    y: np.ndarray, p: np.ndarray, amount: np.ndarray, threshold: float, costs: CostAssumptions
) -> dict[str, float]:
    y = np.asarray(y).astype(bool)
    p = np.asarray(p)
    amount = np.asarray(amount, dtype="float64")
    approved = p < threshold
    n = len(y)
    n_approved = int(approved.sum())
    loss = costs.loss_given_default * amount[y & approved].sum()
    lost_margin = costs.margin_good_loan * amount[~y & ~approved].sum()
    approve_all_cost = costs.loss_given_default * amount[y].sum()
    return {
        "threshold": float(threshold),
        "n": n,
        "approval_rate": n_approved / n if n else float("nan"),
        "default_rate_approved": float(y[approved].mean()) if n_approved else float("nan"),
        "default_rate_rejected": float(y[~approved].mean()) if n_approved < n else float("nan"),
        "defaults_caught": float((y & ~approved).sum() / y.sum()) if y.sum() else float("nan"),
        "good_rejected": float((~y & ~approved).sum() / (~y).sum()) if (~y).sum() else float("nan"),
        "cost_default_losses": float(loss),
        "cost_lost_margin": float(lost_margin),
        "cost_total": float(loss + lost_margin),
        "cost_per_application": float((loss + lost_margin) / n) if n else float("nan"),
        "cost_approve_all": float(approve_all_cost),
        "saving_vs_approve_all": float(1 - (loss + lost_margin) / approve_all_cost)
        if approve_all_cost
        else float("nan"),
    }


def cost_curve(
    y: np.ndarray,
    p: np.ndarray,
    amount: np.ndarray,
    costs: CostAssumptions,
    grid: np.ndarray = DEFAULT_GRID,
) -> pd.DataFrame:
    return pd.DataFrame([outcomes_at(y, p, amount, t, costs) for t in grid])


def choose_threshold(
    y: np.ndarray,
    p: np.ndarray,
    amount: np.ndarray,
    costs: CostAssumptions,
    grid: np.ndarray = DEFAULT_GRID,
) -> tuple[float, pd.DataFrame]:
    """Threshold with the lowest total expected cost on (out-of-fold) predictions."""
    curve = cost_curve(y, p, amount, costs, grid)
    best = curve.loc[curve["cost_total"].idxmin(), "threshold"]
    return float(best), curve
