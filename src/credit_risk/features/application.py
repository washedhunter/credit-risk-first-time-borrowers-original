"""Application-table cleaning and ratio features.

`add_ratio_features` is shared by training and the API, so a feature is always computed the same
way in both places. This module only needs numpy + pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENTINEL_DAYS = 365243  # Home Credit's placeholder for "no date" (e.g. DAYS_EMPLOYED of pensioners)

CATEGORICAL = [
    "NAME_CONTRACT_TYPE",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]

NUMERIC = [
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE",
    "YEARS_EMPLOYED",
    "YEARS_REGISTRATION",
    "YEARS_ID_PUBLISH",
    "YEARS_SINCE_PHONE_CHANGE",
    "OWN_CAR_AGE",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "APARTMENTS_AVG",
    "BASEMENTAREA_AVG",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG",
    "COMMONAREA_AVG",
    "ELEVATORS_AVG",
    "ENTRANCES_AVG",
    "FLOORSMAX_AVG",
    "FLOORSMIN_AVG",
    "LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAREA_AVG",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAREA_AVG",
    "TOTALAREA_MODE",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "FLAG_DOCUMENT_3",
    "DOCUMENT_COUNT",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]

APPLICATION_FEATURES = CATEGORICAL + NUMERIC

# Derived by add_ratio_features (never sent by API clients; always recomputed).
RATIO_FEATURES = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "ANNUITY_CREDIT_RATIO",
    "CREDIT_GOODS_RATIO",
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_MIN",
    "EXT_SOURCE_COUNT",
    "BUREAU_DEBT_INCOME_RATIO",
    "TOTAL_ANNUITY_INCOME_RATIO",
    "ANNUITY_VS_LAST_PREV_RATIO",
]


def _days_to_years(days: pd.Series) -> pd.Series:
    days = days.astype("float64").where(days != SENTINEL_DAYS)
    return (-days / 365.25).round(4)


def clean_application(app: pd.DataFrame) -> pd.DataFrame:
    """Raw application rows -> cleaned base features (plus SK_ID_CURR, and TARGET when present).

    Protected attributes are not returned here; build.py keeps the audit columns separately.
    """
    cols: dict[str, object] = {"SK_ID_CURR": app["SK_ID_CURR"].to_numpy()}
    if "TARGET" in app:
        cols["TARGET"] = app["TARGET"].to_numpy()

    for col in CATEGORICAL:
        values = pd.Series(app[col].to_numpy(), dtype="object")
        cols[col] = (
            values.where(values.notna(), None).replace({"XNA": None, "XAP": None}).to_numpy()
        )

    for flag in ("FLAG_OWN_CAR", "FLAG_OWN_REALTY"):
        cols[flag] = app[flag].map({"Y": 1.0, "N": 0.0}).to_numpy()

    cols["YEARS_EMPLOYED"] = _days_to_years(app["DAYS_EMPLOYED"]).to_numpy()
    cols["YEARS_REGISTRATION"] = _days_to_years(app["DAYS_REGISTRATION"]).to_numpy()
    cols["YEARS_ID_PUBLISH"] = _days_to_years(app["DAYS_ID_PUBLISH"]).to_numpy()
    cols["YEARS_SINCE_PHONE_CHANGE"] = _days_to_years(app["DAYS_LAST_PHONE_CHANGE"]).to_numpy()

    doc_cols = [c for c in app.columns if c.startswith("FLAG_DOCUMENT_")]
    cols["DOCUMENT_COUNT"] = app[doc_cols].sum(axis=1).astype("float64").to_numpy()

    for col in NUMERIC:
        if col not in cols:
            cols[col] = app[col].astype("float64").to_numpy()

    out = pd.DataFrame(cols)
    # Income outliers (one applicant reports 117M) are kept: trees split on rank, and the ratio
    # features below are what carry the signal.
    return out


def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    a = a.astype("float64")
    b = b.astype("float64")
    return (a / b.where(b != 0)).replace([np.inf, -np.inf], np.nan)


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with RATIO_FEATURES added. Missing inputs give NaN."""
    df = df.copy()

    def col(name: str) -> pd.Series:
        return df[name] if name in df else pd.Series(np.nan, index=df.index)

    income = col("AMT_INCOME_TOTAL")
    df["CREDIT_INCOME_RATIO"] = _div(col("AMT_CREDIT"), income)
    df["ANNUITY_INCOME_RATIO"] = _div(col("AMT_ANNUITY"), income)
    df["ANNUITY_CREDIT_RATIO"] = _div(col("AMT_ANNUITY"), col("AMT_CREDIT"))
    df["CREDIT_GOODS_RATIO"] = _div(col("AMT_CREDIT"), col("AMT_GOODS_PRICE"))
    ext = pd.concat([col("EXT_SOURCE_1"), col("EXT_SOURCE_2"), col("EXT_SOURCE_3")], axis=1)
    ext = ext.astype("float64")
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)
    df["EXT_SOURCE_COUNT"] = ext.notna().sum(axis=1).astype("float64")
    df["BUREAU_DEBT_INCOME_RATIO"] = _div(col("BUR_ACTIVE_DEBT_SUM"), income)
    other_annuity = col("BUR_ACTIVE_ANNUITY_SUM").astype("float64").fillna(0.0)
    df["TOTAL_ANNUITY_INCOME_RATIO"] = _div(
        col("AMT_ANNUITY").astype("float64") + other_annuity, income
    )
    df["ANNUITY_VS_LAST_PREV_RATIO"] = _div(col("AMT_ANNUITY"), col("PREV_LAST_AMT_ANNUITY"))
    return df
