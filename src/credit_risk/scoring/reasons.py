"""Turn per-feature SHAP values into plain-English reason codes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from credit_risk.scoring.feature_info import GROUPS, format_value, info

# When the most influential feature in a group is missing, the reason is the absence itself
# (e.g. "insufficient credit history"), which is the typical case for first-time borrowers.
MISSING_SENTENCES = {
    "bureau_history": "No credit-bureau record: no loans with other lenders on file",
    "bureau_arrears": "No credit-bureau record: no loans with other lenders on file",
    "hc_applications": "No earlier Home Credit applications on file",
    "hc_repayment": "No repayment history on earlier Home Credit loans",
    "card_usage": "No Home Credit credit-card history",
    "external_scores": "External credit score not available",
}


@dataclass
class Reason:
    code: str  # reason group key, e.g. "external_scores"
    text: str  # plain-English sentence
    impact: float  # summed SHAP (log-odds) of the group; > 0 raises risk
    top_feature: str
    top_feature_label: str
    top_feature_value: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def reason_codes(
    contributions: Mapping[str, float], values: Mapping[str, Any], top_n: int = 3
) -> tuple[list[Reason], list[Reason]]:
    """Return (reasons that raise risk, strengths that lower it), top_n each, largest first.

    SHAP values are summed per reason group; the group's sentence names its single most
    influential feature (largest contribution in the group's direction) and that feature's value.
    """
    totals: dict[str, float] = defaultdict(float)
    members: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for name, phi in contributions.items():
        group = info(name).group
        totals[group] += phi
        members[group].append((name, phi))

    def build(group: str, total: float) -> Reason:
        sign = 1.0 if total > 0 else -1.0
        name, _ = max(members[group], key=lambda m: sign * m[1])
        label, risk_sentence = GROUPS[group]
        feature_label = info(name).label
        raw = values.get(name)
        value = format_value(name, raw)
        missing = value == "not available"
        if total > 0 and missing and group in MISSING_SENTENCES:
            text = f"{MISSING_SENTENCES[group]}."
        elif total > 0 and missing:
            text = f"{risk_sentence} ({feature_label}: not provided)."
        elif total > 0:
            text = f"{risk_sentence} ({feature_label}: {value})."
        elif missing:
            text = f"{label}: nothing adverse on file, which lowered the risk ({feature_label}: not provided)."
        else:
            text = f"{label} lowered the risk ({feature_label}: {value})."
        return Reason(group, text, round(total, 4), name, feature_label, value)

    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    risks = [build(g, t) for g, t in ranked if t > 0][:top_n]
    strengths = [build(g, t) for g, t in reversed(ranked) if t < 0][:top_n]
    return risks, strengths
