"""Pydantic request/response models. The applicant model is generated from the trained model's
feature list, so the API always accepts exactly what the model was trained on."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from credit_risk.scoring.feature_info import GROUPS, info

REQUIRED = {"AMT_INCOME_TOTAL", "AMT_CREDIT"}
MAX_BATCH = 1000


def _bounds(name: str, kind: str) -> dict[str, float]:
    if name in REQUIRED:
        return {"gt": 0}
    if kind in ("flag", "score"):
        return {"ge": 0, "le": 1}
    if kind in ("count", "years"):
        return {"ge": 0}
    if name.startswith("AMT_"):  # application amounts (bureau/card balances can be negative)
        return {"ge": 0}
    return {}


class ApplicantBase(BaseModel):
    """Cross-field checks: history aggregates must agree with the history counts."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "description": (
                "Application fields plus the applicant's history aggregates (leave history fields "
                "empty for a first-time borrower). Ratio features are computed by the server."
            )
        },
    )

    @model_validator(mode="after")
    def _history_consistent(self) -> ApplicantBase:
        data = self.model_dump()
        for prefix, total_name in (("BUR_", "BUR_COUNT"), ("PREV_", "PREV_COUNT")):
            total = data.get(total_name)
            given = [
                k
                for k, v in data.items()
                if k.startswith(prefix) and k != total_name and v is not None
            ]
            if given and not total:
                raise ValueError(
                    f"{total_name} must be at least 1 when other {prefix}* fields are given "
                    f"(got {total_name}={total} with {given[0]})"
                )
            for k in given:
                if k.endswith("_COUNT") and total is not None and data[k] > total:
                    raise ValueError(f"{k} ({data[k]:g}) cannot exceed {total_name} ({total:g})")
        return self


def build_applicant_model(meta: dict[str, Any]) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    stats = meta.get("feature_stats", {})
    for name in meta["input_features"]:
        fi = info(name)
        desc = f"{fi.label}. Reason group: {GROUPS[fi.group][0]}."
        median = stats.get(name, {}).get("p50")
        example = median if median is not None else None
        if name in meta["categories"]:
            levels = tuple(meta["categories"][name])
            annotation: Any = Literal[levels] | None  # type: ignore[valid-type]
            example = stats.get(name, {}).get("top", [None])[0]
            fields[name] = (annotation, Field(None, description=desc, examples=[example]))
        else:
            required = name in REQUIRED
            annotation = float if required else float | None
            default = ... if required else None
            fields[name] = (
                annotation,
                Field(
                    default,
                    description=desc
                    + ("" if required else " Leave empty if unknown or no history."),
                    allow_inf_nan=False,
                    examples=[example],
                    **_bounds(name, fi.kind),
                ),
            )
    return create_model("Applicant", __base__=ApplicantBase, **fields)


class ReasonOut(BaseModel):
    code: str = Field(description="Reason group key")
    text: str = Field(description="Plain-English reason")
    impact: float = Field(description="Summed SHAP value of the group (log-odds); > 0 raises risk")
    top_feature: str
    top_feature_label: str
    top_feature_value: str


class Explanation(BaseModel):
    base_value: float = Field(
        description="Model's average output in log-odds (SHAP expected value)"
    )
    contributions: dict[str, float] = Field(description="TreeSHAP value per feature, log-odds")
    values: dict[str, Any] = Field(description="Model inputs including the server-computed ratios")


class Prediction(BaseModel):
    probability: float = Field(description="Predicted probability of default", ge=0, le=1)
    decision: Literal["approve", "decline"]
    threshold: float
    thin_file: bool = Field(description="No bureau record and no previous Home Credit application")
    reasons: list[ReasonOut] = Field(description="Top factors raising the risk (up to 3)")
    strengths: list[ReasonOut] = Field(description="Top factors lowering the risk (up to 3)")
    model_version: str
    warnings: list[str] = Field(default_factory=list, description="Data-quality warnings")
    explanation: Explanation | None = None


class BatchPrediction(BaseModel):
    model_version: str
    predictions: list[Prediction]


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str


class FeatureSchema(BaseModel):
    name: str
    label: str
    group: str
    group_label: str
    kind: str
    required: bool
    derived: bool
    categories: list[str] | None = None
    stats: dict[str, Any] = {}


class SampleApplicant(BaseModel):
    id: str
    segment: str
    applicant: dict[str, Any]
