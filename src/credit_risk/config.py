"""Paths and settings. Every value can be overridden with an environment variable (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


COMPETITION = "home-credit-default-risk"

# The seven tables the model uses (Kaggle's application_test has no labels, so it isn't downloaded).
RAW_TABLES = [
    "application_train",
    "bureau",
    "bureau_balance",
    "previous_application",
    "installments_payments",
    "POS_CASH_balance",
    "credit_card_balance",
]

# Protected / sensitive attributes. They are never model features. CODE_GENDER and DAYS_BIRTH are
# kept aside (not as features) only to audit approval rates by group after the fact.
EXCLUDED_FEATURES = [
    "CODE_GENDER",  # gender
    "DAYS_BIRTH",  # age
    "NAME_FAMILY_STATUS",  # marital status
    "CNT_CHILDREN",  # family status
    "CNT_FAM_MEMBERS",  # family status
]
AUDIT_COLUMNS = ["CODE_GENDER", "DAYS_BIRTH"]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: _env_path("DATA_DIR", PROJECT_ROOT / "data"))
    models_dir: Path = field(
        default_factory=lambda: _env_path("MODELS_DIR", PROJECT_ROOT / "models")
    )
    reports_dir: Path = field(
        default_factory=lambda: _env_path("REPORTS_DIR", PROJECT_ROOT / "reports")
    )
    img_dir: Path = field(
        default_factory=lambda: _env_path("IMG_DIR", PROJECT_ROOT / "docs" / "img")
    )
    seed: int = field(default_factory=lambda: _env_int("SEED", 42))
    holdout_fraction: float = field(default_factory=lambda: _env_float("HOLDOUT_FRACTION", 0.2))
    n_folds: int = field(default_factory=lambda: _env_int("N_FOLDS", 5))
    # Expected-cost assumptions, both as a fraction of the loan amount (AMT_CREDIT).
    # Illustrative, not Home Credit's real economics; change them and re-run `credit-risk train`.
    loss_given_default: float = field(
        default_factory=lambda: _env_float("LOSS_GIVEN_DEFAULT", 0.45)
    )
    margin_good_loan: float = field(default_factory=lambda: _env_float("MARGIN_GOOD_LOAN", 0.10))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def sample_dir(self) -> Path:
        return self.data_dir / "sample"

    @property
    def features_path(self) -> Path:
        return self.processed_dir / "features.parquet"


def get_settings() -> Settings:
    return Settings()
