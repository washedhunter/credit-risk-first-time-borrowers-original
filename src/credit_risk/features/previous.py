"""Previous Home Credit applications -> one row per applicant."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.features.application import SENTINEL_DAYS

DAY_COLUMNS = [
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
]


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num = num.astype("float64")
    den = den.astype("float64")
    return (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)


def previous_features(prev: pd.DataFrame) -> pd.DataFrame:
    p = prev.copy()
    for col in DAY_COLUMNS:
        p[col] = p[col].astype("float64").where(p[col] != SENTINEL_DAYS)

    status = p["NAME_CONTRACT_STATUS"]
    approved = status.eq("Approved")
    p["is_approved"] = approved.astype("int32")
    p["is_refused"] = status.eq("Refused").astype("int32")
    p["is_canceled"] = status.isin(["Canceled", "Unused offer"]).astype("int32")
    p["is_recent"] = (p["DAYS_DECISION"] >= -365).astype("int32")
    p["is_high_yield"] = p["NAME_YIELD_GROUP"].eq("high").astype("int32")
    p["is_cash"] = p["NAME_CONTRACT_TYPE"].eq("Cash loans").astype("int32")
    p["is_revolving"] = p["NAME_CONTRACT_TYPE"].eq("Revolving loans").astype("int32")
    # Still running: approved and the planned last due date is in the future.
    p["is_active"] = (approved & (p["DAYS_LAST_DUE_1ST_VERSION"] > 0)).astype("int32")
    p["refused_days"] = p["DAYS_DECISION"].where(status.eq("Refused"))
    p["app_credit_ratio"] = _ratio(p["AMT_APPLICATION"], p["AMT_CREDIT"])
    # Simple-interest proxy: how much more than the principal the instalments add up to, per period.
    total_paid = p["AMT_ANNUITY"].astype("float64") * p["CNT_PAYMENT"].astype("float64")
    p["interest_proxy"] = _ratio(_ratio(total_paid, p["AMT_CREDIT"]) - 1.0, p["CNT_PAYMENT"]).where(
        approved
    )
    for col in ("AMT_CREDIT", "AMT_ANNUITY", "CNT_PAYMENT"):
        p[f"approved_{col}"] = p[col].astype("float64").where(approved)

    g = p.groupby("SK_ID_CURR")
    out = g.agg(
        PREV_COUNT=("SK_ID_PREV", "size"),
        PREV_APPROVED_COUNT=("is_approved", "sum"),
        PREV_REFUSED_COUNT=("is_refused", "sum"),
        PREV_CANCELED_COUNT=("is_canceled", "sum"),
        PREV_RECENT_1Y_COUNT=("is_recent", "sum"),
        PREV_ACTIVE_COUNT=("is_active", "sum"),
        PREV_CASH_COUNT=("is_cash", "sum"),
        PREV_REVOLVING_COUNT=("is_revolving", "sum"),
        high_yield=("is_high_yield", "sum"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
        PREV_DAYS_DECISION_MIN=("DAYS_DECISION", "min"),
        PREV_DAYS_LAST_REFUSED=("refused_days", "max"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_AMT_CREDIT_MEAN=("approved_AMT_CREDIT", "mean"),
        PREV_AMT_CREDIT_MAX=("approved_AMT_CREDIT", "max"),
        PREV_AMT_ANNUITY_MEAN=("approved_AMT_ANNUITY", "mean"),
        PREV_CNT_PAYMENT_MEAN=("approved_CNT_PAYMENT", "mean"),
        PREV_APP_CREDIT_RATIO_MEAN=("app_credit_ratio", "mean"),
        PREV_RATE_DOWN_PAYMENT_MEAN=("RATE_DOWN_PAYMENT", "mean"),
        PREV_INTEREST_PROXY_MEAN=("interest_proxy", "mean"),
        PREV_DAYS_LAST_DUE_MAX=("DAYS_LAST_DUE_1ST_VERSION", "max"),
        PREV_DAYS_TERMINATION_MAX=("DAYS_TERMINATION", "max"),
    )
    out["PREV_APPROVED_RATE"] = _ratio(out["PREV_APPROVED_COUNT"], out["PREV_COUNT"])
    out["PREV_REFUSED_RATE"] = _ratio(out["PREV_REFUSED_COUNT"], out["PREV_COUNT"])
    out["PREV_HIGH_YIELD_RATE"] = _ratio(out["high_yield"], out["PREV_COUNT"])
    out = out.drop(columns=["high_yield"])

    # The most recent application's annuity and status.
    last = p.sort_values(["SK_ID_CURR", "DAYS_DECISION"]).groupby("SK_ID_CURR").tail(1)
    last = last.set_index("SK_ID_CURR")
    out["PREV_LAST_AMT_ANNUITY"] = last["AMT_ANNUITY"].astype("float64")
    out["PREV_LAST_REFUSED"] = last["NAME_CONTRACT_STATUS"].eq("Refused").astype("float64")
    return out.astype("float64")
