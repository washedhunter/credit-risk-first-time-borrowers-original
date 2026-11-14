"""Turn feature records into the exact frame the LightGBM model was trained on.

Used by both training and serving, so categorical encoding and derived ratios can't drift apart.
Only numpy + pandas.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.features.application import add_ratio_features


def prepare_frame(
    df: pd.DataFrame,
    features: Sequence[str],
    categories: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Select `features` in order and give categorical columns their training-time levels.

    Unknown categories become missing, which LightGBM routes like any other missing value.
    """
    columns = {}
    for name in features:
        if name in categories:
            levels = list(categories[name])
            values = df[name] if name in df else pd.Series(None, index=df.index, dtype="object")
            values = values.astype("object").where(values.isin(levels), None)  # unknown -> missing
            columns[name] = pd.Categorical(values, categories=levels)
        else:
            values = df[name] if name in df else pd.Series(np.nan, index=df.index)
            columns[name] = pd.to_numeric(values, errors="coerce").astype("float64")
    out = pd.DataFrame(columns, index=df.index)
    return out


def records_to_frame(
    records: Iterable[Mapping[str, Any]],
    features: Sequence[str],
    categories: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """API path: raw input records -> derived ratios recomputed -> model frame."""
    raw = pd.DataFrame.from_records(list(records))
    for name in features:
        if name not in raw:
            raw[name] = None if name in categories else np.nan
    raw = add_ratio_features(raw)
    return prepare_frame(raw, features, categories)
