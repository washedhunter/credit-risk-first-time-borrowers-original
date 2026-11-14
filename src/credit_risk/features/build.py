"""Join every per-applicant aggregate onto the cleaned application table."""

from __future__ import annotations

import time
from collections.abc import Callable

import pandas as pd

from credit_risk.config import AUDIT_COLUMNS, EXCLUDED_FEATURES, Settings
from credit_risk.features.application import (
    APPLICATION_FEATURES,
    CATEGORICAL,
    RATIO_FEATURES,
    add_ratio_features,
    clean_application,
)
from credit_risk.features.bureau import bureau_features
from credit_risk.features.previous import previous_features
from credit_risk.features.repayments import (
    credit_card_features,
    installments_features,
    pos_cash_features,
)

ID = "SK_ID_CURR"
TARGET = "TARGET"
# Segment flags (not model features): kept in the processed table for reporting.
SEGMENT_COLUMNS = ["THIN_FILE", "NO_BUREAU"]

Tables = dict[str, pd.DataFrame]


def build_features(tables: Tables, log: Callable[[str], None] = print) -> pd.DataFrame:
    """tables: raw frames keyed by table name (see config.RAW_TABLES)."""
    app_raw = tables["application_train"]
    started = time.perf_counter()

    def step(name: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        t0 = time.perf_counter()
        frame = fn()
        log(
            f"  {name}: {frame.shape[1]} features for {len(frame):,} applicants ({time.perf_counter() - t0:.0f}s)"
        )
        return frame

    base = clean_application(app_raw)
    parts = [
        step("bureau", lambda: bureau_features(tables["bureau"], tables.get("bureau_balance"))),
        step("previous", lambda: previous_features(tables["previous_application"])),
        step("installments", lambda: installments_features(tables["installments_payments"])),
        step("pos_cash", lambda: pos_cash_features(tables["POS_CASH_balance"])),
        step("credit_card", lambda: credit_card_features(tables["credit_card_balance"])),
    ]
    base = base.set_index(ID)
    df = pd.concat([base, *(part.reindex(base.index) for part in parts)], axis=1)
    df = df.reset_index().copy()

    for col in ("BUR_COUNT", "PREV_COUNT"):
        df[col] = df[col].fillna(0.0)
    segments = pd.DataFrame(
        {
            "NO_BUREAU": (df["BUR_COUNT"] == 0).astype("int8"),
            "THIN_FILE": ((df["BUR_COUNT"] == 0) & (df["PREV_COUNT"] == 0)).astype("int8"),
        }
    )
    df = pd.concat([df, segments], axis=1)

    df = add_ratio_features(df)

    audit = pd.DataFrame(
        {f"AUDIT_{c}": app_raw[c].to_numpy() for c in AUDIT_COLUMNS}, index=df.index
    )
    df = pd.concat([df, audit], axis=1)

    leaked = [c for c in EXCLUDED_FEATURES if c in feature_columns(df)]
    if leaked:
        raise AssertionError(f"Protected attributes reached the feature set: {leaked}")
    log(
        f"  total: {len(feature_columns(df))} features, {len(df):,} rows ({time.perf_counter() - started:.0f}s)"
    )
    return df


def history_feature_columns(df: pd.DataFrame) -> list[str]:
    prefixes = ("BUR_", "PREV_", "INST_", "POS_", "CC_")
    return [c for c in df.columns if c.startswith(prefixes) and c not in RATIO_FEATURES]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model features, in a stable order: application, history aggregates, derived ratios."""
    return (
        [c for c in APPLICATION_FEATURES if c in df.columns]
        + history_feature_columns(df)
        + [c for c in RATIO_FEATURES if c in df.columns]
    )


def input_feature_columns(df: pd.DataFrame) -> list[str]:
    """What an API client sends: everything except the derived ratios."""
    return [c for c in feature_columns(df) if c not in RATIO_FEATURES]


def categorical_columns() -> list[str]:
    return list(CATEGORICAL)


def load_tables(settings: Settings) -> Tables:
    from credit_risk.config import RAW_TABLES

    missing = [t for t in RAW_TABLES if not (settings.interim_dir / f"{t}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {settings.interim_dir}. Run `uv run credit-risk download` first."
        )
    return {t: pd.read_parquet(settings.interim_dir / f"{t}.parquet") for t in RAW_TABLES}


def build_and_save(settings: Settings, log: Callable[[str], None] = print) -> pd.DataFrame:
    log("Loading tables")
    tables = load_tables(settings)
    log("Building features")
    df = build_features(tables, log=log)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.features_path, index=False)
    log(f"Wrote {settings.features_path}")
    return df
