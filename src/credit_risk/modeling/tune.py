"""Bounded hyper-parameter search with Optuna (fixed trial budget, training split only)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from credit_risk.modeling.lgbm import BASE_PARAMS, cross_validate

TUNE_LEARNING_RATE = 0.1  # faster trials; the final model trains at BASE_PARAMS' 0.05
TUNE_FOLDS = 3


def suggest(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
    }


def tune(
    X: pd.DataFrame,
    y: np.ndarray,
    n_trials: int,
    seed: int,
    out_path: Path,
    log: Callable[[str], None] = print,
) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {**BASE_PARAMS, **suggest(trial), "learning_rate": TUNE_LEARNING_RATE}
        result = cross_validate(
            X,
            y,
            params,
            n_folds=TUNE_FOLDS,
            seed=seed,
            log=lambda _: None,
            early_stopping_rounds=100,
        )
        auc = float(np.mean(result.fold_auc))
        log(f"  trial {trial.number + 1}/{n_trials}: CV AUC {auc:.4f} ({result.seconds:.0f}s)")
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = {
        "params": study.best_params,
        "best_cv_auc_at_lr_0.1": study.best_value,
        "n_trials": n_trials,
        "folds": TUNE_FOLDS,
        "seed": seed,
        "optuna_version": optuna.__version__,
        "tuned_at": datetime.now(UTC).strftime("%Y-%m-%d"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8", newline="\n")
    log(f"Best CV AUC {study.best_value:.4f} -> {out_path}")
    return best
