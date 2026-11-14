"""Repayment records on earlier Home Credit loans: instalments, POS/cash and credit-card balances."""

from __future__ import annotations

import numpy as np
import pandas as pd

UNDERPAY_TOLERANCE = 1.0  # currency units; ignores rounding differences


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num = num.astype("float64")
    den = den.astype("float64")
    return (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)


def installments_features(inst: pd.DataFrame) -> pd.DataFrame:
    """Late-payment rate, underpayment ratio, missed instalments, recency."""
    # One instalment can be paid in several rows (part payments): collapse to one row per instalment.
    keys = ["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"]
    per = inst.groupby(keys, sort=False).agg(
        due_day=("DAYS_INSTALMENT", "first"),
        due_amt=("AMT_INSTALMENT", "first"),
        paid_day=("DAYS_ENTRY_PAYMENT", "max"),
        paid_amt=("AMT_PAYMENT", "sum"),
    )
    per = per.reset_index()
    days_late = (per["paid_day"] - per["due_day"]).astype("float64")
    per["days_late"] = days_late.clip(lower=0)
    per["days_early"] = (-days_late).clip(lower=0)
    per["is_late"] = (days_late > 0).astype("int32")
    per["is_late_30"] = (days_late > 30).astype("int32")
    per["is_underpaid"] = (per["paid_amt"] < per["due_amt"] - UNDERPAY_TOLERANCE).astype("int32")
    per["is_missed"] = (per["paid_amt"].fillna(0) <= 0).astype("int32")
    recent = per["due_day"] >= -365
    per["late_recent"] = per["is_late"].where(recent)
    per["due_recent"] = recent.astype("int32")

    g = per.groupby("SK_ID_CURR")
    out = g.agg(
        INST_COUNT=("due_day", "size"),
        INST_LOANS=("SK_ID_PREV", "nunique"),
        late=("is_late", "sum"),
        late_30=("is_late_30", "sum"),
        under=("is_underpaid", "sum"),
        INST_MISSED_COUNT=("is_missed", "sum"),
        INST_DAYS_LATE_MEAN=("days_late", "mean"),
        INST_DAYS_LATE_MAX=("days_late", "max"),
        INST_DAYS_EARLY_MEAN=("days_early", "mean"),
        due_sum=("due_amt", "sum"),
        paid_sum=("paid_amt", "sum"),
        late_recent=("late_recent", "sum"),
        due_recent=("due_recent", "sum"),
        INST_DAYS_LAST_DUE=("due_day", "max"),
    )
    out["INST_LATE_RATE"] = _ratio(out["late"], out["INST_COUNT"])
    out["INST_LATE_30_RATE"] = _ratio(out["late_30"], out["INST_COUNT"])
    out["INST_UNDERPAID_RATE"] = _ratio(out["under"], out["INST_COUNT"])
    # Share of the total amount due that wasn't paid (0 = paid in full, can be < 0 if overpaid).
    out["INST_UNDERPAYMENT_RATIO"] = 1.0 - _ratio(out["paid_sum"], out["due_sum"])
    out["INST_LATE_RATE_1Y"] = _ratio(out["late_recent"], out["due_recent"])
    out = out.drop(
        columns=["late", "late_30", "under", "due_sum", "paid_sum", "late_recent", "due_recent"]
    )
    return out.astype("float64")


def pos_cash_features(pos: pd.DataFrame) -> pd.DataFrame:
    p = pos.copy()
    p["is_late"] = (p["SK_DPD_DEF"] > 0).astype("int32")
    p["late_recent"] = p["is_late"].where(p["MONTHS_BALANCE"] >= -12)
    p["months_recent"] = (p["MONTHS_BALANCE"] >= -12).astype("int32")
    g = p.groupby("SK_ID_CURR")
    out = g.agg(
        POS_MONTHS=("MONTHS_BALANCE", "size"),
        POS_LOANS=("SK_ID_PREV", "nunique"),
        POS_DPD_MAX=("SK_DPD", "max"),
        POS_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        late=("is_late", "sum"),
        late_recent=("late_recent", "sum"),
        months_recent=("months_recent", "sum"),
    )
    out["POS_LATE_MONTH_RATE"] = _ratio(out["late"], out["POS_MONTHS"])
    out["POS_LATE_RATE_1Y"] = _ratio(out["late_recent"], out["months_recent"])

    latest = p.sort_values(["SK_ID_PREV", "MONTHS_BALANCE"]).groupby("SK_ID_PREV").tail(1)
    latest = latest.assign(
        is_active=latest["NAME_CONTRACT_STATUS"].eq("Active").astype("int32"),
        is_completed=latest["NAME_CONTRACT_STATUS"].eq("Completed").astype("int32"),
    )
    latest_g = latest.groupby("SK_ID_CURR")
    out["POS_REMAINING_INSTALMENTS"] = latest_g["CNT_INSTALMENT_FUTURE"].sum(min_count=1)
    out["POS_ACTIVE_COUNT"] = latest_g["is_active"].sum()
    out["POS_COMPLETED_COUNT"] = latest_g["is_completed"].sum()
    out = out.drop(columns=["late", "late_recent", "months_recent"])
    return out.astype("float64")


def credit_card_features(cc: pd.DataFrame) -> pd.DataFrame:
    c = cc.copy()
    limit = c["AMT_CREDIT_LIMIT_ACTUAL"].astype("float64")
    c["utilisation"] = _ratio(c["AMT_BALANCE"], limit)
    c["is_late"] = (c["SK_DPD_DEF"] > 0).astype("int32")
    min_due = c["AMT_INST_MIN_REGULARITY"].astype("float64")
    has_min = min_due > 0
    c["below_min"] = (c["AMT_PAYMENT_CURRENT"].fillna(0) < min_due).where(has_min).astype("float64")
    c["pay_to_min"] = _ratio(c["AMT_PAYMENT_CURRENT"], min_due)
    c["recent_util"] = c["utilisation"].where(c["MONTHS_BALANCE"] >= -3)

    g = c.groupby("SK_ID_CURR")
    out = g.agg(
        CC_LOANS=("SK_ID_PREV", "nunique"),
        CC_MONTHS=("MONTHS_BALANCE", "size"),
        CC_UTILISATION_MEAN=("utilisation", "mean"),
        CC_UTILISATION_MAX=("utilisation", "max"),
        CC_UTILISATION_3M=("recent_util", "mean"),
        CC_DPD_MAX=("SK_DPD", "max"),
        late=("is_late", "sum"),
        CC_BELOW_MIN_PAYMENT_RATE=("below_min", "mean"),
        CC_PAYMENT_TO_MIN_MEAN=("pay_to_min", "mean"),
        CC_DRAWINGS_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
        CC_ATM_DRAWINGS_MEAN=("AMT_DRAWINGS_ATM_CURRENT", "mean"),
        CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
    )
    out["CC_LATE_MONTH_RATE"] = _ratio(out["late"], out["CC_MONTHS"])
    latest = c[c["MONTHS_BALANCE"] == c.groupby("SK_ID_PREV")["MONTHS_BALANCE"].transform("max")]
    latest_g = latest.groupby("SK_ID_CURR")
    out["CC_BALANCE_LAST"] = latest_g["AMT_BALANCE"].sum(min_count=1)
    out["CC_LIMIT_LAST"] = latest_g["AMT_CREDIT_LIMIT_ACTUAL"].sum(min_count=1)
    out = out.drop(columns=["late"])
    return out.astype("float64")
