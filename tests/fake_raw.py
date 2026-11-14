"""Small fake Home Credit tables with the real schema (randomly generated, for tests only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.features.application import CATEGORICAL, NUMERIC

LEVELS = {
    "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
    "NAME_TYPE_SUITE": ["Unaccompanied", "Family", "Spouse, partner"],
    "NAME_INCOME_TYPE": ["Working", "Pensioner", "Commercial associate", "State servant"],
    "NAME_EDUCATION_TYPE": [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
    ],
    "NAME_HOUSING_TYPE": ["House / apartment", "With parents", "Rented apartment"],
    "OCCUPATION_TYPE": ["Laborers", "Sales staff", "Core staff", "Drivers"],
    "ORGANIZATION_TYPE": ["Business Entity Type 3", "Self-employed", "XNA", "Medicine"],
}


def make_raw_tables(n: int = 3000, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(100001, 100001 + n)
    risk = rng.normal(size=n)  # latent risk drives the target and several features
    target = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-2.6 + 1.2 * risk)))).astype(int)

    app = pd.DataFrame({"SK_ID_CURR": ids, "TARGET": target})
    for col in CATEGORICAL:
        app[col] = rng.choice(LEVELS[col], size=n)
    app["CODE_GENDER"] = rng.choice(["F", "M"], size=n)
    app["DAYS_BIRTH"] = -rng.integers(21 * 365, 68 * 365, size=n)
    app["NAME_FAMILY_STATUS"] = rng.choice(["Married", "Single / not married"], size=n)
    app["CNT_CHILDREN"] = rng.integers(0, 3, size=n)
    app["CNT_FAM_MEMBERS"] = app["CNT_CHILDREN"] + 1.0
    app["FLAG_OWN_CAR"] = rng.choice(["Y", "N"], size=n)
    app["FLAG_OWN_REALTY"] = rng.choice(["Y", "N"], size=n)
    employed = -rng.integers(100, 10000, size=n).astype(float)
    employed[rng.uniform(size=n) < 0.15] = 365243
    app["DAYS_EMPLOYED"] = employed
    app["DAYS_REGISTRATION"] = -rng.integers(0, 15000, size=n).astype(float)
    app["DAYS_ID_PUBLISH"] = -rng.integers(0, 6000, size=n)
    app["DAYS_LAST_PHONE_CHANGE"] = -rng.integers(0, 3000, size=n).astype(float)
    for k in range(2, 22):
        app[f"FLAG_DOCUMENT_{k}"] = (rng.uniform(size=n) < (0.7 if k == 3 else 0.02)).astype(int)
    for col in NUMERIC:
        derived = (
            "YEARS_EMPLOYED",
            "YEARS_REGISTRATION",
            "YEARS_ID_PUBLISH",
            "YEARS_SINCE_PHONE_CHANGE",
            "DOCUMENT_COUNT",
        )
        if col in app.columns or col in derived:
            continue
        app[col] = rng.uniform(0, 1, size=n)
    app["AMT_INCOME_TOTAL"] = rng.lognormal(12, 0.5, size=n)
    app["AMT_CREDIT"] = rng.lognormal(13, 0.6, size=n)
    app["AMT_ANNUITY"] = app["AMT_CREDIT"] / rng.uniform(10, 40, size=n)
    app["AMT_GOODS_PRICE"] = app["AMT_CREDIT"] * rng.uniform(0.8, 1.0, size=n)
    for i, w in ((1, 0.6), (2, 0.9), (3, 0.8)):
        ext = 1 / (1 + np.exp(w * risk + rng.normal(scale=0.8, size=n)))
        ext[rng.uniform(size=n) < (0.5 if i == 1 else 0.1)] = np.nan
        app[f"EXT_SOURCE_{i}"] = ext

    # ~85% have bureau rows, ~95% previous applications; some have neither (thin-file).
    has_bureau = rng.uniform(size=n) < 0.85
    has_prev = rng.uniform(size=n) < 0.95
    has_prev[:40] = False
    has_bureau[:40] = False  # guarantee a thin-file group

    # bureau + bureau_balance
    b_rows = []
    for cid, r, hb in zip(ids, risk, has_bureau, strict=True):
        if not hb:
            continue
        for _ in range(rng.integers(1, 5)):
            active = rng.uniform() < 0.4
            credit = rng.lognormal(12, 1)
            b_rows.append(
                {
                    "SK_ID_CURR": cid,
                    "SK_ID_BUREAU": len(b_rows) + 5000000,
                    "CREDIT_ACTIVE": "Active"
                    if active
                    else rng.choice(["Closed", "Closed", "Sold"]),
                    "CREDIT_CURRENCY": "currency 1",
                    "DAYS_CREDIT": -int(rng.integers(1, 2900)),
                    "CREDIT_DAY_OVERDUE": int(max(0, rng.normal(5 * r, 10)))
                    if rng.uniform() < 0.1
                    else 0,
                    "DAYS_CREDIT_ENDDATE": float(rng.integers(-2000, 2000)),
                    "DAYS_ENDDATE_FACT": np.nan if active else float(-rng.integers(1, 2000)),
                    "AMT_CREDIT_MAX_OVERDUE": float(rng.exponential(1000))
                    if rng.uniform() < 0.3
                    else np.nan,
                    "CNT_CREDIT_PROLONG": int(rng.uniform() < 0.05),
                    "AMT_CREDIT_SUM": credit,
                    "AMT_CREDIT_SUM_DEBT": credit * rng.uniform(0, 1) if active else 0.0,
                    "AMT_CREDIT_SUM_LIMIT": 0.0,
                    "AMT_CREDIT_SUM_OVERDUE": 0.0,
                    "CREDIT_TYPE": rng.choice(
                        ["Consumer credit", "Credit card", "Car loan", "Microloan"]
                    ),
                    "DAYS_CREDIT_UPDATE": -int(rng.integers(0, 1000)),
                    "AMT_ANNUITY": float(rng.exponential(5000)) if active else np.nan,
                }
            )
    bureau = pd.DataFrame(b_rows)
    bb_rows = []
    for bid, cid in zip(bureau["SK_ID_BUREAU"], bureau["SK_ID_CURR"], strict=True):
        r = risk[cid - 100001]
        for m in range(-int(rng.integers(1, 15)), 1):
            late = rng.uniform() < 0.03 * (1 + max(r, 0))
            bb_rows.append(
                {
                    "SK_ID_BUREAU": bid,
                    "MONTHS_BALANCE": m,
                    "STATUS": "1" if late else rng.choice(["0", "C", "X"]),
                }
            )
    bureau_balance = pd.DataFrame(bb_rows)

    # previous applications + repayments
    p_rows, i_rows, pos_rows, cc_rows = [], [], [], []
    for cid, r, hp in zip(ids, risk, has_prev, strict=True):
        if not hp:
            continue
        for _ in range(rng.integers(1, 4)):
            pid = 2000000 + len(p_rows)
            status = rng.choice(["Approved", "Approved", "Refused", "Canceled"])
            credit = float(rng.lognormal(11, 0.8))
            cnt = float(rng.choice([6, 12, 24]))
            p_rows.append(
                {
                    "SK_ID_PREV": pid,
                    "SK_ID_CURR": cid,
                    "NAME_CONTRACT_TYPE": rng.choice(
                        ["Cash loans", "Consumer loans", "Revolving loans"]
                    ),
                    "AMT_ANNUITY": credit / cnt * 1.1,
                    "AMT_APPLICATION": credit * rng.uniform(0.9, 1.1),
                    "AMT_CREDIT": credit,
                    "AMT_GOODS_PRICE": credit,
                    "RATE_DOWN_PAYMENT": float(rng.uniform(0, 0.2)),
                    "NAME_CONTRACT_STATUS": status,
                    "DAYS_DECISION": -int(rng.integers(1, 2900)),
                    "NAME_YIELD_GROUP": rng.choice(["low_normal", "middle", "high", "XNA"]),
                    "CNT_PAYMENT": cnt,
                    "DAYS_FIRST_DRAWING": 365243.0,
                    "DAYS_FIRST_DUE": float(-rng.integers(1, 2000)),
                    "DAYS_LAST_DUE_1ST_VERSION": float(rng.integers(-1500, 500)),
                    "DAYS_LAST_DUE": 365243.0,
                    "DAYS_TERMINATION": float(-rng.integers(1, 1500)),
                }
            )
            if status != "Approved":
                continue
            for k in range(1, int(cnt) + 1):
                due = -30 * k - int(rng.integers(0, 500))
                late = int(max(0, rng.normal(3 * r, 5)))
                amt = credit / cnt
                i_rows.append(
                    {
                        "SK_ID_PREV": pid,
                        "SK_ID_CURR": cid,
                        "NUM_INSTALMENT_VERSION": 1.0,
                        "NUM_INSTALMENT_NUMBER": k,
                        "DAYS_INSTALMENT": float(due),
                        "DAYS_ENTRY_PAYMENT": float(due + late),
                        "AMT_INSTALMENT": amt,
                        "AMT_PAYMENT": amt
                        * (0.5 if rng.uniform() < 0.05 * (1 + max(r, 0)) else 1.0),
                    }
                )
            for m in range(-int(cnt), 0):
                pos_rows.append(
                    {
                        "SK_ID_PREV": pid,
                        "SK_ID_CURR": cid,
                        "MONTHS_BALANCE": m,
                        "CNT_INSTALMENT": cnt,
                        "CNT_INSTALMENT_FUTURE": float(-m),
                        "NAME_CONTRACT_STATUS": "Completed"
                        if m == -1 and rng.uniform() < 0.5
                        else "Active",
                        "SK_DPD": int(rng.uniform() < 0.05) * 10,
                        "SK_DPD_DEF": int(rng.uniform() < 0.03 * (1 + max(r, 0))) * 5,
                    }
                )
            if rng.uniform() < 0.3:
                limit = float(rng.choice([45000, 90000, 135000]))
                for m in range(-6, 0):
                    cc_rows.append(
                        {
                            "SK_ID_PREV": pid,
                            "SK_ID_CURR": cid,
                            "MONTHS_BALANCE": m,
                            "AMT_BALANCE": limit
                            * float(np.clip(rng.normal(0.4 + 0.2 * r, 0.2), 0, 1.2)),
                            "AMT_CREDIT_LIMIT_ACTUAL": limit,
                            "AMT_DRAWINGS_ATM_CURRENT": float(rng.exponential(3000)),
                            "AMT_DRAWINGS_CURRENT": float(rng.exponential(5000)),
                            "AMT_INST_MIN_REGULARITY": limit * 0.05,
                            "AMT_PAYMENT_CURRENT": limit * float(rng.uniform(0, 0.1)),
                            "SK_DPD": 0,
                            "SK_DPD_DEF": 0,
                        }
                    )
    return {
        "application_train": app,
        "bureau": bureau,
        "bureau_balance": bureau_balance,
        "previous_application": pd.DataFrame(p_rows),
        "installments_payments": pd.DataFrame(i_rows),
        "POS_CASH_balance": pd.DataFrame(pos_rows),
        "credit_card_balance": pd.DataFrame(cc_rows),
    }
