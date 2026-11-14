"""Compare the headline numbers of two metrics.json files (reproducibility check)."""

from __future__ import annotations

from typing import Any

HEADLINE = ["roc_auc", "pr_auc", "ks", "brier"]
DECISION = ["approval_rate", "default_rate_approved"]


def headline_values(m: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {"threshold": m["threshold"]}
    for seg, block in m["holdout"].items():
        for key in HEADLINE:
            out[f"holdout.{seg}.lightgbm.{key}"] = block["lightgbm"][key]
            out[f"holdout.{seg}.baseline.{key}"] = block["baseline"][key]
        for key in DECISION:
            out[f"holdout.{seg}.decision.{key}"] = block["decision"][key]
    return out


def compare(expected: dict, actual: dict, tol: float) -> tuple[list[tuple], bool]:
    a, b = headline_values(expected), headline_values(actual)
    rows = []
    for name, va in a.items():
        vb = b.get(name)
        if vb is None:
            rows.append((name, va, float("nan"), float("nan"), False))
            continue
        # Thin-file has ~500 hold-out applicants, so its metrics get a wider tolerance.
        limit = tol * 4 if ".thin_file." in name else tol
        diff = vb - va
        rows.append((name, va, vb, diff, abs(diff) <= limit))
    return rows, all(r[-1] for r in rows)
