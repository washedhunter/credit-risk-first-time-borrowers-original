import math

import numpy as np
import pandas as pd

from credit_risk.config import EXCLUDED_FEATURES
from credit_risk.features.application import (
    APPLICATION_FEATURES,
    SENTINEL_DAYS,
    add_ratio_features,
    clean_application,
)


def test_clean_application_handles_sentinels_flags_and_xna(raw_tables):
    app = raw_tables["application_train"].head(50).copy()
    app.loc[0, "DAYS_EMPLOYED"] = SENTINEL_DAYS
    app.loc[1, "DAYS_EMPLOYED"] = -730.5
    app.loc[0, "ORGANIZATION_TYPE"] = "XNA"
    app.loc[0, "FLAG_OWN_CAR"] = "Y"
    app.loc[1, "FLAG_OWN_CAR"] = "N"
    out = clean_application(app)
    assert math.isnan(out.loc[0, "YEARS_EMPLOYED"])
    assert out.loc[1, "YEARS_EMPLOYED"] == round(730.5 / 365.25, 4)
    assert pd.isna(out.loc[0, "ORGANIZATION_TYPE"])
    assert out.loc[0, "FLAG_OWN_CAR"] == 1.0 and out.loc[1, "FLAG_OWN_CAR"] == 0.0
    docs = [c for c in app.columns if c.startswith("FLAG_DOCUMENT_")]
    assert (out["DOCUMENT_COUNT"] == app[docs].sum(axis=1)).all()


def test_protected_attributes_never_returned(raw_tables):
    out = clean_application(raw_tables["application_train"].head(10))
    assert not set(EXCLUDED_FEATURES) & set(out.columns)
    assert set(APPLICATION_FEATURES) <= set(out.columns)


def test_ratio_features():
    df = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [100.0, 0.0],
            "AMT_CREDIT": [500.0, 10.0],
            "AMT_ANNUITY": [25.0, 1.0],
            "AMT_GOODS_PRICE": [400.0, np.nan],
            "EXT_SOURCE_1": [0.2, np.nan],
            "EXT_SOURCE_2": [0.4, np.nan],
            "EXT_SOURCE_3": [np.nan, np.nan],
            "BUR_ACTIVE_DEBT_SUM": [50.0, np.nan],
            "BUR_ACTIVE_ANNUITY_SUM": [5.0, np.nan],
            "PREV_LAST_AMT_ANNUITY": [50.0, np.nan],
        }
    )
    out = add_ratio_features(df)
    r = out.iloc[0]
    assert r["CREDIT_INCOME_RATIO"] == 5.0
    assert r["ANNUITY_INCOME_RATIO"] == 0.25
    assert r["ANNUITY_CREDIT_RATIO"] == 0.05
    assert r["CREDIT_GOODS_RATIO"] == 1.25
    assert math.isclose(r["EXT_SOURCE_MEAN"], 0.3) and r["EXT_SOURCE_MIN"] == 0.2
    assert r["EXT_SOURCE_COUNT"] == 2
    assert r["BUREAU_DEBT_INCOME_RATIO"] == 0.5
    assert math.isclose(r["TOTAL_ANNUITY_INCOME_RATIO"], 0.30)
    assert r["ANNUITY_VS_LAST_PREV_RATIO"] == 0.5
    # zero income / missing inputs -> NaN, never inf
    z = out.iloc[1]
    assert math.isnan(z["CREDIT_INCOME_RATIO"]) and math.isnan(z["CREDIT_GOODS_RATIO"])
    assert math.isnan(z["EXT_SOURCE_MEAN"]) and z["EXT_SOURCE_COUNT"] == 0
    assert np.isfinite(out.select_dtypes("number").fillna(0).to_numpy()).all()


def test_ratio_features_tolerate_missing_columns():
    out = add_ratio_features(pd.DataFrame({"AMT_INCOME_TOTAL": [1.0], "AMT_CREDIT": [2.0]}))
    assert out.loc[0, "CREDIT_INCOME_RATIO"] == 2.0
    assert math.isnan(out.loc[0, "BUREAU_DEBT_INCOME_RATIO"])
