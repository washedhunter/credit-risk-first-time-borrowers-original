import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from credit_risk.api.app import app

client = TestClient(app)


@pytest.fixture(scope="module")
def samples():
    return client.get("/samples").json()


def test_health_and_model():
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_version"].startswith("lgbm-")
    m = client.get("/model").json()
    assert 0 < m["threshold"] < 1 and m["version"] == h["model_version"]
    assert {"overall", "thin_file"} <= set(m["holdout_metrics"])


def test_schema_lists_every_feature():
    schema = client.get("/schema").json()
    meta = client.get("/model").json()
    assert len(schema) > 100
    by_name = {f["name"]: f for f in schema}
    assert by_name["AMT_CREDIT"]["required"] and not by_name["AMT_CREDIT"]["derived"]
    assert by_name["CREDIT_INCOME_RATIO"]["derived"]
    assert by_name["NAME_INCOME_TYPE"]["categories"]
    assert not set(meta["excluded_features"]) & set(by_name)


def test_samples(samples):
    assert len(samples) == 300
    thin = client.get("/samples", params={"segment": "thin_file", "limit": 5}).json()
    assert len(thin) == 5 and all(s["segment"] == "thin_file" for s in thin)


def test_predict_single(samples):
    r = client.post("/predict", json=samples[0]["applicant"])
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["probability"] < 1
    assert body["decision"] == ("approve" if body["probability"] < body["threshold"] else "decline")
    assert 1 <= len(body["reasons"]) <= 3 and all(x["impact"] > 0 for x in body["reasons"])
    assert all(x["impact"] < 0 for x in body["strengths"])
    assert "explanation" not in body
    assert body["model_version"].startswith("lgbm-")


def test_predict_with_explanation(samples):
    body = client.post("/predict", params={"explain": True}, json=samples[150]["applicant"]).json()
    expl = body["explanation"]
    assert len(expl["contributions"]) > 100
    assert expl["values"]["CREDIT_INCOME_RATIO"] is not None  # computed server-side


def test_batch_matches_single(samples):
    batch = client.post(
        "/predict/batch", json={"applicants": [s["applicant"] for s in samples[:20]]}
    ).json()
    assert len(batch["predictions"]) == 20
    single = client.post("/predict", json=samples[7]["applicant"]).json()
    assert batch["predictions"][7]["probability"] == pytest.approx(single["probability"])


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"AMT_CREDIT": 100000}, "AMT_INCOME_TOTAL"),  # required field missing
        ({"AMT_CREDIT": 100000, "AMT_INCOME_TOTAL": -5}, "greater than 0"),
        ({"AMT_CREDIT": 100000, "AMT_INCOME_TOTAL": 5e4, "CREDIT_INCOME_RATIO": 2}, "Extra inputs"),
        (
            {"AMT_CREDIT": 100000, "AMT_INCOME_TOTAL": 5e4, "NAME_INCOME_TYPE": "Wizard"},
            "Input should be",
        ),
        (
            {"AMT_CREDIT": 100000, "AMT_INCOME_TOTAL": 5e4, "EXT_SOURCE_2": 1.7},
            "less than or equal to 1",
        ),
        ({"AMT_CREDIT": 100000, "AMT_INCOME_TOTAL": 5e4, "CODE_GENDER": "F"}, "Extra inputs"),
        (
            {"AMT_CREDIT": 100000, "AMT_INCOME_TOTAL": 5e4, "AMT_ANNUITY": -5000},
            "greater than or equal to 0",
        ),
        (
            {"AMT_CREDIT": 100000, "AMT_INCOME_TOTAL": 5e4, "BUR_ACTIVE_COUNT": 1},
            "BUR_COUNT must be at least 1",
        ),
        (
            {"AMT_CREDIT": 100000, "AMT_INCOME_TOTAL": 5e4, "BUR_COUNT": 1, "BUR_ACTIVE_COUNT": 3},
            "cannot exceed BUR_COUNT",
        ),
        (
            {
                "AMT_CREDIT": 100000,
                "AMT_INCOME_TOTAL": 5e4,
                "PREV_COUNT": 0,
                "PREV_REFUSED_RATE": 0.5,
            },
            "PREV_COUNT must be at least 1",
        ),
    ],
)
def test_validation_errors(payload, fragment):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert fragment in r.text


def test_rejects_non_finite_numbers():
    r = client.post(
        "/predict",
        content='{"AMT_CREDIT": 100000, "AMT_INCOME_TOTAL": 50000, "AMT_ANNUITY": Infinity}',
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422


def test_batch_limits(samples):
    assert client.post("/predict/batch", json={"applicants": []}).status_code == 422
    too_many = [samples[0]["applicant"]] * 1001
    assert client.post("/predict/batch", json={"applicants": too_many}).status_code == 422


def test_sparse_input_gets_a_warning():
    body = client.post("/predict", json={"AMT_CREDIT": 100000, "AMT_INCOME_TOTAL": 50000}).json()
    assert body["warnings"] and "less reliable" in body["warnings"][0]
    assert body["thin_file"] is True


def test_api_works_without_training_libraries():
    """Simulate the API image, where sklearn/shap/optuna/matplotlib/kaggle aren't installed."""
    code = """
import importlib.abc, sys
BLOCKED = ("sklearn", "shap", "optuna", "matplotlib", "kaggle", "streamlit")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked: {name}")
sys.meta_path.insert(0, Block())
from fastapi.testclient import TestClient
from credit_risk.api.app import app
c = TestClient(app)
s = c.get("/samples", params={"limit": 1}).json()[0]["applicant"]
r = c.post("/predict", params={"explain": True}, json=s)
assert r.status_code == 200, r.text
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0 and "ok" in result.stdout, result.stdout + result.stderr
