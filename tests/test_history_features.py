import math

import pandas as pd
import pytest

from credit_risk.features.bureau import bureau_features
from credit_risk.features.previous import previous_features
from credit_risk.features.repayments import (
    credit_card_features,
    installments_features,
    pos_cash_features,
)


def _bureau_row(**kw):
    base = {
        "SK_ID_CURR": 1,
        "SK_ID_BUREAU": 10,
        "CREDIT_ACTIVE": "Active",
        "DAYS_CREDIT": -100,
        "CREDIT_DAY_OVERDUE": 0,
        "DAYS_CREDIT_ENDDATE": 200.0,
        "DAYS_ENDDATE_FACT": float("nan"),
        "AMT_CREDIT_MAX_OVERDUE": float("nan"),
        "CNT_CREDIT_PROLONG": 0,
        "AMT_CREDIT_SUM": 1000.0,
        "AMT_CREDIT_SUM_DEBT": 400.0,
        "AMT_CREDIT_SUM_LIMIT": 0.0,
        "AMT_CREDIT_SUM_OVERDUE": 0.0,
        "CREDIT_TYPE": "Consumer credit",
        "DAYS_CREDIT_UPDATE": -5,
        "AMT_ANNUITY": 50.0,
    }
    return {**base, **kw}


def test_bureau_aggregates_and_balance():
    bureau = pd.DataFrame(
        [
            _bureau_row(),
            _bureau_row(
                SK_ID_BUREAU=11,
                CREDIT_ACTIVE="Closed",
                DAYS_CREDIT=-900,
                AMT_CREDIT_SUM=2000.0,
                AMT_CREDIT_SUM_DEBT=0.0,
                DAYS_ENDDATE_FACT=-300.0,
                AMT_ANNUITY=float("nan"),
            ),
            _bureau_row(
                SK_ID_BUREAU=12,
                CREDIT_TYPE="Credit card",
                AMT_CREDIT_SUM=500.0,
                AMT_CREDIT_SUM_DEBT=250.0,
                AMT_CREDIT_SUM_LIMIT=250.0,
                CREDIT_DAY_OVERDUE=12,
            ),
            _bureau_row(SK_ID_CURR=2, SK_ID_BUREAU=20, CREDIT_ACTIVE="Bad debt"),
        ]
    )
    bb = pd.DataFrame(
        {
            "SK_ID_BUREAU": [10, 10, 10, 10, 11],
            "MONTHS_BALANCE": [-20, -2, -1, 0, -5],
            "STATUS": ["0", "2", "X", "C", "1"],
        }
    )
    out = bureau_features(bureau, bb)
    a = out.loc[1]
    assert a["BUR_COUNT"] == 3 and a["BUR_ACTIVE_COUNT"] == 2 and a["BUR_CLOSED_COUNT"] == 1
    assert a["BUR_DAYS_CREDIT_MAX"] == -100 and a["BUR_DAYS_CREDIT_MIN"] == -900
    assert a["BUR_ACTIVE_DEBT_SUM"] == 650 and a["BUR_ACTIVE_CREDIT_SUM"] == 1500
    assert a["BUR_ACTIVE_DEBT_RATIO"] == pytest.approx(650 / 1500)
    assert a["BUR_CARD_UTILISATION"] == pytest.approx(250 / 250)
    assert a["BUR_OVERDUE_NOW_COUNT"] == 1 and a["BUR_DAY_OVERDUE_MAX"] == 12
    assert a["BUR_DAYS_LAST_CLOSED"] == -300
    # bureau_balance: 3 known months on loan 10 (X is unknown) + 1 on loan 11; late = "2" and "1"
    assert a["BUR_BB_MONTHS"] == 4
    assert a["BUR_BB_LATE_RATE"] == pytest.approx(2 / 4)
    assert a["BUR_BB_WORST_STATUS"] == 2
    assert a["BUR_BB_LATE_12M_COUNT"] == 2
    assert out.loc[2, "BUR_BAD_DEBT_COUNT"] == 1
    assert math.isnan(out.loc[2, "BUR_BB_MONTHS"])


def test_previous_applications():
    prev = pd.DataFrame(
        {
            "SK_ID_PREV": [1, 2, 3],
            "SK_ID_CURR": [7, 7, 7],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Consumer loans", "Cash loans"],
            "AMT_ANNUITY": [110.0, 50.0, 80.0],
            "AMT_APPLICATION": [1000.0, 600.0, 900.0],
            "AMT_CREDIT": [1000.0, 500.0, 900.0],
            "AMT_GOODS_PRICE": [1000.0, 500.0, 900.0],
            "RATE_DOWN_PAYMENT": [0.0, 0.1, 0.0],
            "NAME_CONTRACT_STATUS": ["Approved", "Approved", "Refused"],
            "DAYS_DECISION": [-400, -100, -10],
            "NAME_YIELD_GROUP": ["high", "low_normal", "XNA"],
            "CNT_PAYMENT": [10.0, 12.0, 12.0],
            "DAYS_FIRST_DRAWING": [365243.0] * 3,
            "DAYS_FIRST_DUE": [-370.0, -70.0, 365243.0],
            "DAYS_LAST_DUE_1ST_VERSION": [-100.0, 300.0, 365243.0],
            "DAYS_LAST_DUE": [365243.0] * 3,
            "DAYS_TERMINATION": [-90.0, 365243.0, 365243.0],
        }
    )
    out = previous_features(prev).loc[7]
    assert (
        out["PREV_COUNT"] == 3
        and out["PREV_APPROVED_COUNT"] == 2
        and out["PREV_REFUSED_COUNT"] == 1
    )
    assert out["PREV_REFUSED_RATE"] == pytest.approx(1 / 3)
    assert out["PREV_ACTIVE_COUNT"] == 1  # approved and the last due date is in the future
    assert out["PREV_DAYS_LAST_REFUSED"] == -10
    assert out["PREV_DAYS_LAST_DUE_MAX"] == 300  # the 365243 sentinel became missing
    assert out["PREV_DAYS_TERMINATION_MAX"] == -90
    assert out["PREV_LAST_REFUSED"] == 1.0 and out["PREV_LAST_AMT_ANNUITY"] == 80.0
    # interest proxy on approved loans: (110*10/1000 - 1)/10 = 0.01 and (50*12/500 - 1)/12
    assert out["PREV_INTEREST_PROXY_MEAN"] == pytest.approx((0.01 + 0.2 / 12) / 2)
    assert out["PREV_AMT_CREDIT_MEAN"] == 750  # approved only
    assert out["PREV_HIGH_YIELD_RATE"] == pytest.approx(1 / 3)


def test_installments_collapse_part_payments():
    inst = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 1],
            "SK_ID_PREV": [9, 9, 9, 9],
            "NUM_INSTALMENT_VERSION": [1.0, 1.0, 1.0, 1.0],
            "NUM_INSTALMENT_NUMBER": [1, 1, 2, 3],
            "DAYS_INSTALMENT": [-90.0, -90.0, -60.0, -30.0],
            "DAYS_ENTRY_PAYMENT": [-95.0, -80.0, -61.0, -30.0],
            "AMT_INSTALMENT": [100.0, 100.0, 100.0, 100.0],
            "AMT_PAYMENT": [60.0, 40.0, 100.0, 50.0],
        }
    )
    out = installments_features(inst).loc[1]
    assert out["INST_COUNT"] == 3  # instalment 1 was paid in two parts
    assert out["INST_LATE_RATE"] == pytest.approx(1 / 3)  # the last part of #1 came 10 days late
    assert out["INST_DAYS_LATE_MAX"] == 10
    assert out["INST_UNDERPAID_RATE"] == pytest.approx(1 / 3)
    assert out["INST_UNDERPAYMENT_RATIO"] == pytest.approx(1 - 250 / 300)
    assert out["INST_MISSED_COUNT"] == 0


def test_pos_and_credit_card():
    pos = pd.DataFrame(
        {
            "SK_ID_PREV": [5, 5, 5, 6],
            "SK_ID_CURR": [1, 1, 1, 1],
            "MONTHS_BALANCE": [-3, -2, -1, -1],
            "CNT_INSTALMENT": [3.0, 3.0, 3.0, 6.0],
            "CNT_INSTALMENT_FUTURE": [2.0, 1.0, 0.0, 4.0],
            "NAME_CONTRACT_STATUS": ["Active", "Active", "Completed", "Active"],
            "SK_DPD": [0, 5, 0, 0],
            "SK_DPD_DEF": [0, 5, 0, 0],
        }
    )
    p = pos_cash_features(pos).loc[1]
    assert p["POS_LATE_MONTH_RATE"] == 0.25 and p["POS_DPD_MAX"] == 5
    assert p["POS_REMAINING_INSTALMENTS"] == 4
    assert p["POS_ACTIVE_COUNT"] == 1 and p["POS_COMPLETED_COUNT"] == 1

    cc = pd.DataFrame(
        {
            "SK_ID_PREV": [3, 3],
            "SK_ID_CURR": [1, 1],
            "MONTHS_BALANCE": [-2, -1],
            "AMT_BALANCE": [50.0, 90.0],
            "AMT_CREDIT_LIMIT_ACTUAL": [100.0, 100.0],
            "AMT_DRAWINGS_ATM_CURRENT": [0.0, 10.0],
            "AMT_DRAWINGS_CURRENT": [0.0, 20.0],
            "AMT_INST_MIN_REGULARITY": [5.0, 5.0],
            "AMT_PAYMENT_CURRENT": [10.0, 2.0],
            "SK_DPD": [0, 0],
            "SK_DPD_DEF": [0, 0],
        }
    )
    c = credit_card_features(cc).loc[1]
    assert c["CC_UTILISATION_MEAN"] == pytest.approx(0.7)
    assert c["CC_UTILISATION_MAX"] == pytest.approx(0.9)
    assert c["CC_BELOW_MIN_PAYMENT_RATE"] == 0.5
    assert c["CC_BALANCE_LAST"] == 90 and c["CC_LIMIT_LAST"] == 100
