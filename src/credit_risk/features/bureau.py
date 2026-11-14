"""Credit-bureau history (other lenders): bureau + bureau_balance -> one row per applicant."""

from __future__ import annotations

import numpy as np
import pandas as pd

LATE_STATUSES = {"1", "2", "3", "4", "5"}  # bureau_balance STATUS: months past due bucket


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num = num.astype("float64")
    den = den.astype("float64")
    return (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)


def bureau_balance_per_loan(bb: pd.DataFrame) -> pd.DataFrame:
    """Monthly status rows -> per bureau loan: months observed, late months, worst status."""
    status = bb["STATUS"].astype("string")
    known = ~status.isin(["X"])  # X = status unknown
    late = status.isin(LATE_STATUSES)
    worst = pd.to_numeric(status.where(late), errors="coerce").fillna(0)
    recent = bb["MONTHS_BALANCE"] >= -12
    frame = pd.DataFrame(
        {
            "SK_ID_BUREAU": bb["SK_ID_BUREAU"],
            "known": known.astype("int32"),
            "late": late.astype("int32"),
            "late_12m": (late & recent).astype("int32"),
            "worst": worst,
        }
    )
    return frame.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS=("known", "sum"),
        BB_LATE_MONTHS=("late", "sum"),
        BB_LATE_12M=("late_12m", "sum"),
        BB_WORST=("worst", "max"),
    )


def bureau_features(bureau: pd.DataFrame, bureau_balance: pd.DataFrame | None) -> pd.DataFrame:
    b = bureau.copy()
    active = b["CREDIT_ACTIVE"].eq("Active")
    b["is_active"] = active.astype("int32")
    b["is_closed"] = b["CREDIT_ACTIVE"].eq("Closed").astype("int32")
    b["is_bad"] = b["CREDIT_ACTIVE"].isin(["Bad debt", "Sold"]).astype("int32")
    b["is_recent"] = (b["DAYS_CREDIT"] >= -365).astype("int32")
    b["is_card"] = b["CREDIT_TYPE"].eq("Credit card").astype("int32")
    b["is_microloan"] = b["CREDIT_TYPE"].eq("Microloan").astype("int32")
    b["is_mortgage_or_car"] = (
        b["CREDIT_TYPE"].isin(["Mortgage", "Car loan", "Real estate loan"]).astype("int32")
    )
    b["is_overdue"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype("int32")
    b["active_debt"] = b["AMT_CREDIT_SUM_DEBT"].where(active).clip(lower=0)
    b["active_credit"] = b["AMT_CREDIT_SUM"].where(active)
    b["active_annuity"] = b["AMT_ANNUITY"].where(active)
    card = b["CREDIT_TYPE"].eq("Credit card") & active
    b["card_debt"] = b["AMT_CREDIT_SUM_DEBT"].where(card).clip(lower=0)
    b["card_limit"] = b["AMT_CREDIT_SUM_LIMIT"].where(card)
    b["card_limit"] = b["card_limit"].where(b["card_limit"] > 0, b["AMT_CREDIT_SUM"].where(card))

    g = b.groupby("SK_ID_CURR")
    out = g.agg(
        BUR_COUNT=("SK_ID_BUREAU", "size"),
        BUR_ACTIVE_COUNT=("is_active", "sum"),
        BUR_CLOSED_COUNT=("is_closed", "sum"),
        BUR_BAD_DEBT_COUNT=("is_bad", "sum"),
        BUR_RECENT_1Y_COUNT=("is_recent", "sum"),
        BUR_CARD_COUNT=("is_card", "sum"),
        BUR_MICROLOAN_COUNT=("is_microloan", "sum"),
        BUR_MORTGAGE_CAR_COUNT=("is_mortgage_or_car", "sum"),
        BUR_OVERDUE_NOW_COUNT=("is_overdue", "sum"),
        BUR_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
        BUR_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
        BUR_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUR_DAYS_ENDDATE_MAX=("DAYS_CREDIT_ENDDATE", "max"),
        BUR_DAYS_UPDATE_MAX=("DAYS_CREDIT_UPDATE", "max"),
        BUR_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUR_AMT_MAX_OVERDUE_MAX=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUR_AMT_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUR_PROLONG_SUM=("CNT_CREDIT_PROLONG", "sum"),
        BUR_AMT_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        BUR_AMT_CREDIT_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUR_ACTIVE_DEBT_SUM=("active_debt", "sum"),
        BUR_ACTIVE_CREDIT_SUM=("active_credit", "sum"),
        BUR_ACTIVE_ANNUITY_SUM=("active_annuity", "sum"),
        card_debt=("card_debt", "sum"),
        card_limit=("card_limit", "sum"),
    )
    out["BUR_ACTIVE_RATE"] = _ratio(out["BUR_ACTIVE_COUNT"], out["BUR_COUNT"])
    out["BUR_ACTIVE_DEBT_RATIO"] = _ratio(out["BUR_ACTIVE_DEBT_SUM"], out["BUR_ACTIVE_CREDIT_SUM"])
    out["BUR_CARD_UTILISATION"] = _ratio(out["card_debt"], out["card_limit"])
    out = out.drop(columns=["card_debt", "card_limit"])

    # Recency of the latest closed loan's actual end (DAYS_ENDDATE_FACT only exists once closed).
    out["BUR_DAYS_LAST_CLOSED"] = g["DAYS_ENDDATE_FACT"].max()

    if bureau_balance is not None and len(bureau_balance):
        per_loan = bureau_balance_per_loan(bureau_balance)
        merged = b[["SK_ID_CURR", "SK_ID_BUREAU"]].merge(
            per_loan, left_on="SK_ID_BUREAU", right_index=True, how="inner"
        )
        bb = merged.groupby("SK_ID_CURR").agg(
            months=("BB_MONTHS", "sum"),
            late=("BB_LATE_MONTHS", "sum"),
            BUR_BB_LATE_12M_COUNT=("BB_LATE_12M", "sum"),
            BUR_BB_WORST_STATUS=("BB_WORST", "max"),
        )
        bb["BUR_BB_MONTHS"] = bb["months"]
        bb["BUR_BB_LATE_RATE"] = _ratio(bb["late"], bb["months"])
        out = out.join(bb.drop(columns=["months", "late"]), how="left")
    else:
        for col in (
            "BUR_BB_LATE_12M_COUNT",
            "BUR_BB_WORST_STATUS",
            "BUR_BB_MONTHS",
            "BUR_BB_LATE_RATE",
        ):
            out[col] = np.nan

    return out.astype("float64")
