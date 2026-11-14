"""Synthetic applicants for the demo, the API tests and CI (no real rows are committed).

A Gaussian copula is fitted per segment on the engineered input features. Every column gets
a *value* latent (normal scores of its within-observed ranks) and, if it is sometimes missing,
a *presence* latent (missing vs present). Correlations between all latents are estimated on
pairwise-complete rows, repaired to the nearest positive-definite matrix, and sampled. Values
are mapped back through each column's interpolated quantile function (101 percentile points),
so they are smoothed population percentiles, never a copy of a real applicant, and features
that go missing together (e.g. all bureau aggregates) stay missing together.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

QUANTILES = np.linspace(0, 1, 101)


def _column_model(s: pd.Series, categorical: bool) -> dict:
    miss = float(s.isna().mean())
    obs = s.dropna()
    if categorical:
        freq = obs.value_counts(normalize=True)
        return {
            "kind": "cat",
            "miss": miss,
            "levels": freq.index.tolist(),
            "probs": freq.to_numpy(),
        }
    values = obs.to_numpy(dtype="float64")
    integer = bool(len(values)) and bool(np.all(np.isclose(values, np.round(values))))
    q = np.quantile(values, QUANTILES) if len(values) else np.array([])
    return {"kind": "num", "miss": miss, "q": q, "integer": integer}


def _value_scores(s: pd.Series, model: dict, rng: np.random.Generator) -> np.ndarray:
    """Normal scores of the observed values (NaN where missing)."""
    z = np.full(len(s), np.nan)
    obs = s.notna().to_numpy()
    if not obs.any():
        return z
    if model["kind"] == "cat":
        order = {lvl: i for i, lvl in enumerate(model["levels"])}
        codes = s[obs].map(order).to_numpy(dtype=int)
        cum = np.concatenate([[0], np.cumsum(model["probs"])])
        u = cum[codes] + rng.uniform(0, 1, len(codes)) * (cum[codes + 1] - cum[codes])
    else:
        u = stats.rankdata(s[obs].to_numpy(), method="average") / (obs.sum() + 1)
    z[obs] = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    return z


def _presence_scores(s: pd.Series, miss: float) -> np.ndarray:
    """Missing -> midpoint of the lower block, present -> midpoint of the upper block."""
    present = s.notna().to_numpy()
    return np.where(present, stats.norm.ppf((1 + miss) / 2), stats.norm.ppf(miss / 2))


def _from_value(z: np.ndarray, model: dict) -> np.ndarray:
    u = stats.norm.cdf(z)
    if model["kind"] == "cat":
        cum = np.cumsum(model["probs"])
        idx = np.minimum(np.searchsorted(cum, u, side="right"), len(cum) - 1)
        return np.array([model["levels"][i] for i in idx], dtype=object)
    vals = np.interp(u, QUANTILES, model["q"])
    return np.round(vals) if model["integer"] else vals


def _nearest_correlation(corr: np.ndarray) -> np.ndarray:
    corr = np.nan_to_num(corr, nan=0.0)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    w, v = np.linalg.eigh(corr)
    corr = (v * np.clip(w, 1e-4, None)) @ v.T
    d = np.sqrt(np.diag(corr))
    return corr / np.outer(d, d)


def fit_sample(
    df: pd.DataFrame, columns: Sequence[str], categorical: Sequence[str], n: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    models = {c: _column_model(df[c], c in categorical) for c in columns}
    latents: dict[str, np.ndarray] = {}
    for c in columns:
        m = models[c]
        if m["miss"] < 1.0:
            latents[f"v:{c}"] = _value_scores(df[c], m, rng)
        if 0.0 < m["miss"] < 1.0:
            latents[f"p:{c}"] = _presence_scores(df[c], m["miss"])
    frame = pd.DataFrame(latents)
    keep = [k for k in frame.columns if frame[k].nunique() > 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = frame[keep].corr(min_periods=30).to_numpy()  # pairwise-complete
    corr = _nearest_correlation(corr)
    draws = pd.DataFrame(
        rng.multivariate_normal(np.zeros(len(keep)), corr, size=n, method="eigh"), columns=keep
    )

    out: dict[str, np.ndarray] = {}
    for c in columns:
        m = models[c]
        if m["miss"] >= 1.0:
            out[c] = np.full(
                n,
                None if m["kind"] == "cat" else np.nan,
                dtype=object if m["kind"] == "cat" else "float64",
            )
            continue
        vkey = f"v:{c}"
        z = draws[vkey].to_numpy() if vkey in draws else np.zeros(n)
        values = _from_value(z, m)
        pkey = f"p:{c}"
        if pkey in draws:
            missing = stats.norm.cdf(draws[pkey].to_numpy()) < m["miss"]
            values = values.astype(object) if m["kind"] == "cat" else values.astype("float64")
            values[missing] = None if m["kind"] == "cat" else np.nan
        out[c] = values
    return pd.DataFrame(out)


def _enforce_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Repair a few logical relations the copula can't guarantee."""
    df = df.copy()
    for prefix in ("BUR", "PREV"):
        total = df[f"{prefix}_COUNT"]
        for col in [c for c in df.columns if c.startswith(f"{prefix}_") and c.endswith("_COUNT")]:
            if col != f"{prefix}_COUNT":
                df[col] = np.minimum(df[col], total)
    shares = [c for c in df.columns if c.endswith(("_RATE", "_RATE_1Y"))]
    df[shares] = df[shares].clip(0, 1)
    for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        df[col] = df[col].clip(0, 1)
    return df


SEGMENT_SIZES = {"thin_file": 60, "no_bureau_has_prev": 60, "has_bureau": 180}


def make_synthetic_sample(
    train: pd.DataFrame,
    input_features: Sequence[str],
    categorical: Sequence[str],
    seed: int,
    sizes: dict[str, int] | None = None,
) -> pd.DataFrame:
    sizes = sizes or SEGMENT_SIZES
    segments = {
        "thin_file": train["THIN_FILE"] == 1,
        "no_bureau_has_prev": (train["NO_BUREAU"] == 1) & (train["THIN_FILE"] == 0),
        "has_bureau": train["NO_BUREAU"] == 0,
    }
    parts = []
    for k, (name, mask) in enumerate(segments.items()):
        part = fit_sample(train.loc[mask], input_features, categorical, sizes[name], seed + k)
        part.insert(0, "SEGMENT", name)
        parts.append(part)
    out = _enforce_consistency(pd.concat(parts, ignore_index=True))
    out.insert(0, "SYNTHETIC_ID", [f"SYN-{i + 1:04d}" for i in range(len(out))])
    return out
