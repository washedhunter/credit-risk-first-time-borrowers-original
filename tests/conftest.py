from __future__ import annotations

import pytest
from fake_raw import make_raw_tables

from credit_risk.features.build import build_features


@pytest.fixture(scope="session")
def raw_tables():
    return make_raw_tables(n=3000, seed=0)


@pytest.fixture(scope="session")
def features(raw_tables):
    return build_features(raw_tables, log=lambda _: None)


@pytest.fixture(autouse=True)
def _no_kaggle(monkeypatch):
    """Tests never talk to Kaggle."""
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
