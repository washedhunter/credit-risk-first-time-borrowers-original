"""Logistic-regression baseline. All preprocessing lives inside the sklearn Pipeline, so it is
re-fitted on the training folds only (no information from the validation fold leaks in)."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer


def make_baseline(numeric: Sequence[str], categorical: Sequence[str], seed: int) -> Pipeline:
    numeric_pipe = make_pipeline(
        # Rank -> normal scores: robust to the heavy-tailed amounts; NaNs pass through untouched.
        QuantileTransformer(
            n_quantiles=200, output_distribution="normal", subsample=100_000, random_state=seed
        ),
        SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True),
    )
    categorical_pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200),
    )
    pre = ColumnTransformer(
        [("num", numeric_pipe, list(numeric)), ("cat", categorical_pipe, list(categorical))]
    )
    model = LogisticRegression(C=0.05, max_iter=2000)
    return Pipeline([("pre", pre), ("model", model)])


def _as_object(X: pd.DataFrame, categorical: Sequence[str]) -> pd.DataFrame:
    X = X.copy()
    for col in categorical:
        X[col] = X[col].astype("object").where(X[col].notna(), np.nan)
    return X


def cross_validate_baseline(
    X: pd.DataFrame,
    y: np.ndarray,
    numeric: Sequence[str],
    categorical: Sequence[str],
    n_folds: int,
    seed: int,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, Pipeline, float]:
    """Return OOF probabilities, a pipeline refitted on all of X, and the seconds taken."""
    started = time.perf_counter()
    X = _as_object(X, categorical)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pipe = make_baseline(numeric, categorical, seed)
    oof = cross_val_predict(pipe, X, y, cv=folds, method="predict_proba", n_jobs=n_folds)[:, 1]
    final = make_baseline(numeric, categorical, seed).fit(X, y)
    seconds = time.perf_counter() - started
    log(f"    logistic baseline: {seconds:.0f}s")
    return oof, final, seconds


def predict_baseline(pipe: Pipeline, X: pd.DataFrame, categorical: Sequence[str]) -> np.ndarray:
    return pipe.predict_proba(_as_object(X, categorical))[:, 1]
