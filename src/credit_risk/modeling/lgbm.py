"""LightGBM training: stratified K-fold with early stopping, and the final refit."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE_PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "min_split_gain": 0.0,
    "cat_smooth": 10.0,
    "verbose": -1,
    "deterministic": True,
    "force_col_wise": True,
}
EARLY_STOPPING_ROUNDS = 200
MAX_ROUNDS = 10_000

# Imbalance strategies compared in training. Values are functions of (negatives / positives).
IMBALANCE_STRATEGIES: dict[str, Callable[[float], float]] = {
    "none": lambda ratio: 1.0,
    "sqrt": lambda ratio: float(np.sqrt(ratio)),
    "balanced": lambda ratio: float(ratio),
}


@dataclass
class CVResult:
    oof: np.ndarray
    best_iterations: list[int]
    fold_auc: list[float]
    seconds: float

    @property
    def mean_best_iteration(self) -> int:
        return round(float(np.mean(self.best_iterations)))


def load_tuned_params(path: Path) -> dict:
    """Optuna's best params (config/lgbm_params.json) on top of BASE_PARAMS."""
    params = dict(BASE_PARAMS)
    if path.exists():
        params.update(json.loads(path.read_text())["params"])
    return params


def cross_validate(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    n_folds: int,
    seed: int,
    log: Callable[[str], None] = print,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    max_rounds: int = MAX_ROUNDS,
) -> CVResult:
    started = time.perf_counter()
    params = {**params, "seed": seed}
    oof = np.zeros(len(y), dtype="float64")
    best_iterations: list[int] = []
    fold_auc: list[float] = []
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(folds.split(X, y), start=1):
        dtrain = lgb.Dataset(X.iloc[tr], label=y[tr], free_raw_data=False)
        dvalid = lgb.Dataset(X.iloc[va], label=y[va], reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=max_rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        oof[va] = booster.predict(X.iloc[va], num_iteration=booster.best_iteration)
        best_iterations.append(int(booster.best_iteration))
        fold_auc.append(float(booster.best_score["valid_0"]["auc"]))
        log(f"    fold {k}: AUC {fold_auc[-1]:.4f} at {best_iterations[-1]} trees")
    return CVResult(oof, best_iterations, fold_auc, time.perf_counter() - started)


def fit_final(
    X: pd.DataFrame, y: np.ndarray, params: dict, num_rounds: int, seed: int
) -> lgb.Booster:
    params = {**params, "seed": seed}
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=num_rounds)
