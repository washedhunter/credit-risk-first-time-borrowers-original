"""Stratified train / hold-out split. The hold-out is touched once, at the very end."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def holdout_split(
    df: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratify on TARGET x THIN_FILE so the small thin-file segment is represented in both parts."""
    strata = df["TARGET"].astype(str) + "_" + df["THIN_FILE"].astype(str)
    train, holdout = train_test_split(df, test_size=fraction, stratify=strata, random_state=seed)
    return train.reset_index(drop=True), holdout.reset_index(drop=True)
