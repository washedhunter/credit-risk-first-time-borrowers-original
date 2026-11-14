import pytest

from credit_risk.config import EXCLUDED_FEATURES
from credit_risk.features.application import RATIO_FEATURES
from credit_risk.features.build import feature_columns, input_feature_columns
from credit_risk.scoring.feature_info import FEATURES


def test_one_row_per_applicant_and_segment_flags(raw_tables, features):
    app = raw_tables["application_train"]
    assert len(features) == len(app)
    assert features["SK_ID_CURR"].is_unique
    in_bureau = app["SK_ID_CURR"].isin(raw_tables["bureau"]["SK_ID_CURR"]).to_numpy()
    in_prev = app["SK_ID_CURR"].isin(raw_tables["previous_application"]["SK_ID_CURR"]).to_numpy()
    assert (features["THIN_FILE"].to_numpy() == (~in_bureau & ~in_prev)).all()
    assert (features["NO_BUREAU"].to_numpy() == ~in_bureau).all()
    thin = features[features["THIN_FILE"] == 1]
    assert len(thin) >= 40
    assert (thin["BUR_COUNT"] == 0).all() and (thin["PREV_COUNT"] == 0).all()
    assert thin["INST_LATE_RATE"].isna().all()


def test_feature_set(features):
    cols = feature_columns(features)
    assert not set(cols) & set(EXCLUDED_FEATURES)
    assert not {"TARGET", "SK_ID_CURR", "THIN_FILE", "NO_BUREAU"} & set(cols)
    assert not any(c.startswith("AUDIT_") for c in cols)
    assert set(cols) == set(FEATURES), "every feature needs a label and a reason group"
    assert set(cols) - set(input_feature_columns(features)) == set(RATIO_FEATURES)


def test_protected_attribute_guard(raw_tables, monkeypatch):
    import credit_risk.features.build as build

    monkeypatch.setattr(build, "feature_columns", lambda df: ["CODE_GENDER"])
    with pytest.raises(AssertionError, match="Protected"):
        build.build_features(raw_tables, log=lambda _: None)


def test_audit_columns_kept_outside_features(features):
    assert {"AUDIT_CODE_GENDER", "AUDIT_DAYS_BIRTH"} <= set(features.columns)
