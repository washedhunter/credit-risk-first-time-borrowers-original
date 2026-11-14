"""Score one example applicant against a running API (standard library only).

uv run python scripts/predict_example.py            # http://localhost:8000
uv run python scripts/predict_example.py http://host:8000
"""

from __future__ import annotations

import json
import sys
import urllib.request

API = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"

# A first-time borrower: application fields only, no credit history.
applicant = {
    "NAME_CONTRACT_TYPE": "Cash loans",
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_HOUSING_TYPE": "House / apartment",
    "OCCUPATION_TYPE": "Sales staff",
    "ORGANIZATION_TYPE": "Trade: type 7",
    "AMT_INCOME_TOTAL": 180000,
    "AMT_CREDIT": 500000,
    "AMT_ANNUITY": 25000,
    "AMT_GOODS_PRICE": 450000,
    "YEARS_EMPLOYED": 3.5,
    "YEARS_REGISTRATION": 10,
    "YEARS_ID_PUBLISH": 6,
    "YEARS_SINCE_PHONE_CHANGE": 1.5,
    "FLAG_OWN_REALTY": 1,
    "FLAG_OWN_CAR": 0,
    "REGION_RATING_CLIENT": 2,
    "REGION_RATING_CLIENT_W_CITY": 2,
    "NAME_TYPE_SUITE": "Unaccompanied",
    "REGION_POPULATION_RELATIVE": 0.02,
    "FLAG_EMP_PHONE": 1,
    "FLAG_WORK_PHONE": 0,
    "FLAG_PHONE": 0,
    "FLAG_EMAIL": 0,
    "REG_REGION_NOT_LIVE_REGION": 0,
    "REG_REGION_NOT_WORK_REGION": 0,
    "LIVE_REGION_NOT_WORK_REGION": 0,
    "REG_CITY_NOT_LIVE_CITY": 0,
    "REG_CITY_NOT_WORK_CITY": 0,
    "LIVE_CITY_NOT_WORK_CITY": 0,
    "OBS_30_CNT_SOCIAL_CIRCLE": 1,
    "DEF_30_CNT_SOCIAL_CIRCLE": 0,
    "OBS_60_CNT_SOCIAL_CIRCLE": 1,
    "DEF_60_CNT_SOCIAL_CIRCLE": 0,
    "FLAG_DOCUMENT_3": 1,
    "DOCUMENT_COUNT": 1,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.5,
    "BUR_COUNT": 0,
    "PREV_COUNT": 0,
}

request = urllib.request.Request(
    f"{API}/predict",
    data=json.dumps(applicant).encode(),
    headers={"content-type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    result = json.load(response)

print(
    f"P(default) = {result['probability']:.3f} -> {result['decision'].upper()} (threshold {result['threshold']:.3f})"
)
print(f"Thin-file: {result['thin_file']}   model: {result['model_version']}")
for reason in result["reasons"]:
    print(f"  - {reason['text']}")
for strength in result["strengths"]:
    print(f"  + {strength['text']}")
for warning in result["warnings"]:
    print(f"  ! {warning}")
