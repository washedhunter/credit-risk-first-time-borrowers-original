"""Load the trained model and score applicants. Serving-only: numpy, pandas, lightgbm."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from credit_risk.scoring.frame import records_to_frame
from credit_risk.scoring.reasons import Reason, reason_codes

MODEL_FILE = "model.txt"
META_FILE = "model_meta.json"


@dataclass
class Score:
    probability: float
    decision: str  # "approve" | "decline"
    thin_file: bool
    reasons: list[Reason]
    strengths: list[Reason]
    base_value: float  # model's average log-odds (SHAP expected value)
    contributions: dict[str, float] = field(default_factory=dict)  # log-odds per feature
    values: dict[str, Any] = field(default_factory=dict)  # model inputs incl. derived ratios
    warnings: list[str] = field(default_factory=list)


class ScoringModel:
    def __init__(self, booster: lgb.Booster, meta: dict[str, Any]):
        self.booster = booster
        self.meta = meta
        self.features: list[str] = meta["features"]
        self.input_features: list[str] = meta["input_features"]
        self.categories: dict[str, list[str]] = meta["categories"]
        self.threshold: float = float(meta["threshold"])
        if booster.feature_name() != self.features:
            raise ValueError("model.txt and model_meta.json disagree on the feature list")
        # Application fields that (almost) every training applicant had. If a request leaves them
        # empty it is outside what the model has seen, so the response carries a warning.
        history = ("BUR_", "PREV_", "INST_", "POS_", "CC_")
        stats = meta.get("feature_stats", {})
        self.expected_fields = [
            f
            for f in self.input_features
            if not f.startswith(history) and stats.get(f, {}).get("missing_rate", 1.0) < 0.01
        ]

    @classmethod
    def load(cls, models_dir: Path) -> ScoringModel:
        booster = lgb.Booster(model_file=str(models_dir / MODEL_FILE))
        meta = json.loads((models_dir / META_FILE).read_text(encoding="utf-8"))
        return cls(booster, meta)

    @property
    def version(self) -> str:
        return str(self.meta["version"])

    def frame(self, records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
        return records_to_frame(records, self.features, self.categories)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """TreeSHAP values in log-odds, shape (n, n_features + 1); the last column is the base value."""
        return np.asarray(self.booster.predict(X, pred_contrib=True))

    def decide(self, probability: float) -> str:
        return "approve" if probability < self.threshold else "decline"

    def score(self, records: list[Mapping[str, Any]], full: bool = False) -> list[Score]:
        X = self.frame(records)
        proba = self.predict_proba(X)
        contrib = self.contributions(X)
        out = []
        for i in range(len(X)):
            row = X.iloc[i]
            values = {name: _plain(row[name]) for name in self.features}
            phi = dict(zip(self.features, contrib[i, :-1].tolist(), strict=True))
            reasons, strengths = reason_codes(phi, values, top_n=3)
            thin = _is_zero_or_missing(values.get("BUR_COUNT")) and _is_zero_or_missing(
                values.get("PREV_COUNT")
            )
            absent = [f for f in self.expected_fields if values.get(f) is None]
            warnings = (
                [
                    f"{len(absent)} field(s) present in over 99% of training applications are empty "
                    f"({', '.join(absent[:8])}{', ...' if len(absent) > 8 else ''}); the score is less reliable."
                ]
                if absent
                else []
            )
            out.append(
                Score(
                    probability=float(proba[i]),
                    decision=self.decide(float(proba[i])),
                    thin_file=thin,
                    reasons=reasons,
                    strengths=strengths,
                    base_value=float(contrib[i, -1]),
                    contributions=phi if full else {},
                    values=values if full else {},
                    warnings=warnings,
                )
            )
        return out


def _plain(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _is_zero_or_missing(value: Any) -> bool:
    return value is None or value == 0
