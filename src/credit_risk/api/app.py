"""FastAPI service: `uvicorn credit_risk.api.app:app` (or `docker compose up`)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from credit_risk.api.schemas import (
    MAX_BATCH,
    REQUIRED,
    BatchPrediction,
    Explanation,
    FeatureSchema,
    Health,
    Prediction,
    ReasonOut,
    SampleApplicant,
    build_applicant_model,
)
from credit_risk.config import PROJECT_ROOT
from credit_risk.scoring.feature_info import GROUPS, info
from credit_risk.scoring.model import Score, ScoringModel

MODELS_DIR = Path(os.environ.get("MODELS_DIR", PROJECT_ROOT / "models"))
SAMPLE_PATH = Path(
    os.environ.get("SAMPLE_PATH", PROJECT_ROOT / "data" / "sample" / "synthetic_applicants.csv")
)

model = ScoringModel.load(MODELS_DIR)
Applicant = build_applicant_model(model.meta)


def _load_samples(path: Path) -> list[SampleApplicant]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    out = []
    for row in df.to_dict(orient="records"):
        applicant = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in row.items()
            if k in model.input_features
        }
        out.append(
            SampleApplicant(id=row["SYNTHETIC_ID"], segment=row["SEGMENT"], applicant=applicant)
        )
    return out


SAMPLES = _load_samples(SAMPLE_PATH)


app = FastAPI(
    title="Credit risk scoring for first-time borrowers",
    version=model.version,
    description=(
        "LightGBM default-risk model trained on the Home Credit Default Risk dataset. Returns the "
        "probability of default, an approve/decline decision against a cost-tuned threshold, and "
        "plain-English reason codes from TreeSHAP. Sample applicants are **synthetic**."
    ),
)


def _json_safe(obj: Any) -> Any:
    """Validation errors echo the input; NaN/Infinity there would make the error itself invalid JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))}
    )


def _to_prediction(s: Score, explain: bool) -> Prediction:
    return Prediction(
        probability=s.probability,
        decision=s.decision,  # type: ignore[arg-type]
        threshold=model.threshold,
        thin_file=s.thin_file,
        reasons=[ReasonOut(**r.as_dict()) for r in s.reasons],
        strengths=[ReasonOut(**r.as_dict()) for r in s.strengths],
        model_version=model.version,
        warnings=s.warnings,
        explanation=Explanation(
            base_value=s.base_value, contributions=s.contributions, values=s.values
        )
        if explain
        else None,
    )


@app.get("/health", response_model=Health, tags=["service"])
def health() -> Health:
    return Health(status="ok", model_version=model.version)


@app.get("/model", tags=["service"])
def model_info() -> dict[str, Any]:
    """Model version, threshold, cost assumptions and hold-out metrics."""
    keys = [
        "version",
        "trained_at",
        "threshold",
        "decision_rule",
        "cost_assumptions",
        "imbalance_strategy",
        "num_boost_round",
        "thin_file_definition",
        "excluded_features",
        "holdout_metrics",
        "base_default_rate",
    ]
    return {k: model.meta.get(k) for k in keys}


@app.get("/schema", response_model=list[FeatureSchema], tags=["service"])
def schema() -> list[FeatureSchema]:
    """Every model feature with its label, reason group, type and training-data percentiles."""
    derived = set(model.meta["derived_features"])
    out = []
    for name in model.features:
        fi = info(name)
        out.append(
            FeatureSchema(
                name=name,
                label=fi.label,
                group=fi.group,
                group_label=GROUPS[fi.group][0],
                kind=fi.kind,
                required=name in REQUIRED,
                derived=name in derived,
                categories=model.categories.get(name),
                stats=model.meta["feature_stats"].get(name, {}),
            )
        )
    return out


@app.get("/samples", response_model=list[SampleApplicant], tags=["service"])
def samples(
    segment: Annotated[
        str | None, Query(description="thin_file | no_bureau_has_prev | has_bureau")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 1000,
) -> list[SampleApplicant]:
    """Synthetic sample applicants (not real people) for demos and testing."""
    rows = [s for s in SAMPLES if segment is None or s.segment == segment]
    return rows[:limit]


@app.post("/predict", response_model=Prediction, response_model_exclude_none=True, tags=["scoring"])
def predict(
    applicant: Annotated[Applicant, Body()],  # type: ignore[valid-type]
    explain: Annotated[bool, Query(description="Include every feature's SHAP value")] = False,
) -> Prediction:
    """Score one applicant."""
    score = model.score([applicant.model_dump()], full=explain)[0]
    return _to_prediction(score, explain)


@app.post(
    "/predict/batch",
    response_model=BatchPrediction,
    response_model_exclude_none=True,
    tags=["scoring"],
)
def predict_batch(
    applicants: Annotated[list[Applicant], Body(embed=True)],  # type: ignore[valid-type]
) -> BatchPrediction:
    """Score up to 1,000 applicants in one call."""
    if not 1 <= len(applicants) <= MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"Send between 1 and {MAX_BATCH} applicants.")
    scores = model.score([a.model_dump() for a in applicants])
    return BatchPrediction(
        model_version=model.version, predictions=[_to_prediction(s, False) for s in scores]
    )
