"""End-to-end training: split -> baseline -> LightGBM (imbalance comparison) -> threshold ->
hold-out evaluation -> SHAP -> model + reports + synthetic sample."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from credit_risk.config import AUDIT_COLUMNS, EXCLUDED_FEATURES, PROJECT_ROOT, Settings
from credit_risk.features.application import CATEGORICAL, RATIO_FEATURES
from credit_risk.features.build import build_and_save, feature_columns, input_feature_columns
from credit_risk.modeling import explain, plots
from credit_risk.modeling.baseline import cross_validate_baseline, predict_baseline
from credit_risk.modeling.lgbm import (
    IMBALANCE_STRATEGIES,
    cross_validate,
    fit_final,
    load_tuned_params,
)
from credit_risk.modeling.metrics import bootstrap_ci, summarize
from credit_risk.modeling.report import write_reports
from credit_risk.modeling.split import holdout_split
from credit_risk.modeling.threshold import (
    CostAssumptions,
    choose_threshold,
    cost_curve,
    outcomes_at,
)
from credit_risk.scoring.frame import prepare_frame
from credit_risk.scoring.model import META_FILE, MODEL_FILE
from credit_risk.synth import make_synthetic_sample

PARAMS_PATH = PROJECT_ROOT / "config" / "lgbm_params.json"
SEGMENTS = ("overall", "thin_file", "no_bureau")
SENSITIVITY = [(lgd, m) for lgd in (0.30, 0.45, 0.60) for m in (0.05, 0.10, 0.20)]


def _segment_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "overall": np.ones(len(df), dtype=bool),
        "thin_file": df["THIN_FILE"].to_numpy() == 1,
        "no_bureau": df["NO_BUREAU"].to_numpy() == 1,
    }


def _feature_stats(train: pd.DataFrame, features: list[str]) -> dict[str, dict]:
    """Per-feature ranges for the API schema, the dashboard form and the what-if sliders."""
    stats: dict[str, dict] = {}
    for name in features:
        s = train[name]
        entry: dict = {"missing_rate": round(float(s.isna().mean()), 4)}
        if s.notna().sum() == 0:
            stats[name] = entry
            continue
        if name in CATEGORICAL:
            entry["top"] = s.value_counts().head(1).index.tolist()
        else:
            q = s.quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
            entry.update(
                {
                    f"p{round(k * 100):02d}": (None if pd.isna(v) else round(float(v), 6))
                    for k, v in q.items()
                }
            )
        stats[name] = entry
    return stats


def _audit(holdout: pd.DataFrame, p: np.ndarray, threshold: float) -> dict[str, list[dict]]:
    """Approval and bad rates by gender and age band (attributes never used as features)."""
    frame = pd.DataFrame(
        {
            "y": holdout["TARGET"].to_numpy(),
            "p": p,
            "approved": p < threshold,
            "gender": holdout["AUDIT_CODE_GENDER"].to_numpy(),
            "age": -holdout["AUDIT_DAYS_BIRTH"].to_numpy() / 365.25,
        }
    )
    frame["age_band"] = pd.cut(
        frame["age"], [0, 30, 45, 60, 120], labels=["<30", "30-44", "45-59", "60+"], right=False
    )
    out: dict[str, list[dict]] = {}
    for col in ("gender", "age_band"):
        rows = []
        for key, g in frame.groupby(col, observed=True):
            if len(g) < 50:
                continue
            rows.append(
                {
                    "group": str(key),
                    "n": len(g),
                    "default_rate": float(g["y"].mean()),
                    "mean_score": float(g["p"].mean()),
                    "approval_rate": float(g["approved"].mean()),
                    "default_rate_approved": float(g.loc[g["approved"], "y"].mean()),
                }
            )
        out[col] = rows
    return out


def train(
    settings: Settings, rebuild_features: bool = False, log: Callable[[str], None] = print
) -> dict:
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    costs = CostAssumptions(settings.loss_given_default, settings.margin_good_loan)

    if rebuild_features or not settings.features_path.exists():
        t0 = time.perf_counter()
        df = build_and_save(settings, log=log)
        timings["features_s"] = time.perf_counter() - t0
    else:
        log(f"Using cached features {settings.features_path}")
        df = pd.read_parquet(settings.features_path)

    features = feature_columns(df)
    input_features = input_feature_columns(df)
    assert not set(features) & set(EXCLUDED_FEATURES)

    train_df, holdout = holdout_split(df, settings.holdout_fraction, settings.seed)
    categories = {c: sorted(train_df[c].dropna().unique().tolist()) for c in CATEGORICAL}
    X_tr = prepare_frame(train_df, features, categories)
    X_ho = prepare_frame(holdout, features, categories)
    y_tr = train_df["TARGET"].to_numpy()
    y_ho = holdout["TARGET"].to_numpy()
    amt_tr = train_df["AMT_CREDIT"].to_numpy()
    amt_ho = holdout["AMT_CREDIT"].to_numpy()
    seg_tr = _segment_masks(train_df)
    seg_ho = _segment_masks(holdout)
    log(f"Split: train {len(train_df):,} / hold-out {len(holdout):,}; {len(features)} features")

    # --- logistic baseline
    log("Logistic-regression baseline (5-fold OOF + refit)")
    numeric = [f for f in features if f not in CATEGORICAL]
    t0 = time.perf_counter()
    oof_lr, lr_model, _ = cross_validate_baseline(
        X_tr, y_tr, numeric, CATEGORICAL, settings.n_folds, settings.seed, log=log
    )
    p_ho_lr = predict_baseline(lr_model, X_ho, CATEGORICAL)
    timings["baseline_s"] = time.perf_counter() - t0

    # --- LightGBM: compare imbalance strategies on OOF predictions
    params = load_tuned_params(PARAMS_PATH)
    pos_ratio = float((y_tr == 0).sum() / (y_tr == 1).sum())
    strategies: dict[str, dict] = {}
    oofs: dict[str, np.ndarray] = {}
    cv_results = {}
    t0 = time.perf_counter()
    for name, weight_fn in IMBALANCE_STRATEGIES.items():
        spw = weight_fn(pos_ratio)
        log(f"LightGBM 5-fold CV, scale_pos_weight={spw:.2f} ({name})")
        res = cross_validate(
            X_tr,
            y_tr,
            {**params, "scale_pos_weight": spw},
            settings.n_folds,
            settings.seed,
            log=log,
        )
        thr, curve = choose_threshold(y_tr, res.oof, amt_tr, costs)
        best = curve.loc[curve["threshold"] == thr].iloc[0]
        strategies[name] = {
            "scale_pos_weight": spw,
            **summarize(y_tr, res.oof),
            "fold_auc": res.fold_auc,
            "best_iterations": res.best_iterations,
            "best_threshold": thr,
            "min_cost_per_application": float(best["cost_per_application"]),
            "approval_rate_at_best": float(best["approval_rate"]),
            "seconds": res.seconds,
        }
        oofs[name] = res.oof
        cv_results[name] = res
    timings["lgbm_cv_s"] = time.perf_counter() - t0
    chosen = min(strategies, key=lambda k: strategies[k]["min_cost_per_application"])
    log(f"Chosen imbalance strategy: {chosen} (lowest out-of-fold expected cost)")

    oof = oofs[chosen]
    threshold, oof_curve = choose_threshold(y_tr, oof, amt_tr, costs)
    n_rounds = cv_results[chosen].mean_best_iteration
    final_params = {**params, "scale_pos_weight": strategies[chosen]["scale_pos_weight"]}

    t0 = time.perf_counter()
    booster = fit_final(X_tr, y_tr, final_params, n_rounds, settings.seed)
    timings["final_fit_s"] = time.perf_counter() - t0
    p_ho = booster.predict(X_ho)

    # --- hold-out evaluation
    log("Hold-out evaluation")
    evaluation: dict[str, dict] = {}
    for seg in SEGMENTS:
        m = seg_ho[seg]
        evaluation[seg] = {
            "lightgbm": {
                **summarize(y_ho[m], p_ho[m]),
                "ci95": bootstrap_ci(y_ho[m], p_ho[m], seed=settings.seed),
            },
            "baseline": {
                **summarize(y_ho[m], p_ho_lr[m]),
                "ci95": bootstrap_ci(y_ho[m], p_ho_lr[m], seed=settings.seed),
            },
            "decision": outcomes_at(y_ho[m], p_ho[m], amt_ho[m], threshold, costs),
        }
    oof_segments = {
        seg: {
            "lightgbm": {
                **summarize(y_tr[seg_tr[seg]], oof[seg_tr[seg]]),
                "ci95": bootstrap_ci(y_tr[seg_tr[seg]], oof[seg_tr[seg]], seed=settings.seed),
            },
            "baseline": summarize(y_tr[seg_tr[seg]], oof_lr[seg_tr[seg]]),
        }
        for seg in SEGMENTS
    }

    # Segment-specific thresholds (chosen on OOF) - discussed, not deployed.
    segment_thresholds = {}
    for seg in ("thin_file", "no_bureau"):
        m = seg_tr[seg]
        t_seg, _ = choose_threshold(y_tr[m], oof[m], amt_tr[m], costs)
        segment_thresholds[seg] = {
            "oof_optimal_threshold": t_seg,
            "holdout_at_segment_threshold": outcomes_at(
                y_ho[seg_ho[seg]], p_ho[seg_ho[seg]], amt_ho[seg_ho[seg]], t_seg, costs
            ),
        }

    sensitivity = []
    for lgd, margin in SENSITIVITY:
        c = CostAssumptions(lgd, margin)
        t, _ = choose_threshold(y_tr, oof, amt_tr, c)
        o = outcomes_at(y_ho, p_ho, amt_ho, t, c)
        sensitivity.append(
            {
                "loss_given_default": lgd,
                "margin_good_loan": margin,
                "threshold": t,
                "approval_rate": o["approval_rate"],
                "default_rate_approved": o["default_rate_approved"],
                "saving_vs_approve_all": o["saving_vs_approve_all"],
            }
        )

    audit = _audit(holdout, p_ho, threshold)

    # --- SHAP
    log("SHAP explanations")
    t0 = time.perf_counter()
    img = settings.img_dir
    rng = np.random.default_rng(settings.seed)
    idx = np.sort(rng.choice(len(X_ho), size=min(5000, len(X_ho)), replace=False))
    X_shap = X_ho.iloc[idx]
    expl = explain.shap_values(booster, X_shap)
    native = booster.predict(X_shap.iloc[:200], pred_contrib=True)
    shap_gap = float(np.max(np.abs(native[:, :-1] - expl.values[:200])))
    explain.summary_plots(expl, img)
    by_feature, by_group = explain.importance_tables(expl)
    plots.group_importance(by_group, img / "shap_reason_groups.png")
    p_shap = p_ho[idx]
    declined = np.where((p_shap >= threshold) & (holdout["THIN_FILE"].to_numpy()[idx] == 0))[0]
    example = int(declined[len(declined) // 2]) if len(declined) else 0
    explain.waterfall(
        expl,
        example,
        img / "shap_waterfall.png",
        title=f"One declined hold-out applicant: P(default) = {p_shap[example]:.2f} (threshold {threshold:.3f})",
    )
    timings["shap_s"] = time.perf_counter() - t0

    # --- plots
    plots.roc_pr(
        {
            "baseline": (y_ho, p_ho_lr),
            "overall": (y_ho, p_ho),
            "no_bureau": (y_ho[seg_ho["no_bureau"]], p_ho[seg_ho["no_bureau"]]),
            "thin_file": (y_ho[seg_ho["thin_file"]], p_ho[seg_ho["thin_file"]]),
        },
        img / "roc_pr.png",
    )
    plots.calibration(
        {
            "overall": (y_ho, p_ho, 10),
            "thin_file": (y_ho[seg_ho["thin_file"]], p_ho[seg_ho["thin_file"]], 5),
        },
        {k: (y_tr, v) for k, v in oofs.items()},
        img / "calibration.png",
    )
    plots.cost_vs_threshold(
        {
            "Train, out-of-fold (used to choose)": oof_curve,
            "Hold-out, all applicants": cost_curve(y_ho, p_ho, amt_ho, costs),
            "Hold-out, thin-file": cost_curve(
                y_ho[seg_ho["thin_file"]],
                p_ho[seg_ho["thin_file"]],
                amt_ho[seg_ho["thin_file"]],
                costs,
            ),
        },
        threshold,
        img / "cost_threshold.png",
        breakeven=costs.breakeven_probability,
    )

    # --- save model + metadata
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = settings.models_dir / MODEL_FILE
    booster.save_model(str(model_path))
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:8]
    trained_at = datetime.now(UTC).strftime("%Y-%m-%d")
    version = f"lgbm-{trained_at}-{digest}"
    meta = {
        "version": version,
        "trained_at": trained_at,
        "features": features,
        "input_features": input_features,
        "derived_features": [f for f in features if f in RATIO_FEATURES],
        "categorical_features": CATEGORICAL,
        "categories": categories,
        "excluded_features": EXCLUDED_FEATURES,
        "threshold": threshold,
        "decision_rule": "approve if P(default) < threshold, else decline",
        "cost_assumptions": costs.as_dict(),
        "imbalance_strategy": chosen,
        "scale_pos_weight": final_params["scale_pos_weight"],
        "num_boost_round": n_rounds,
        "params": final_params,
        "thin_file_definition": "no credit-bureau records and no previous Home Credit applications",
        "holdout_metrics": {
            seg: {
                k: evaluation[seg]["lightgbm"][k] for k in ("n", "roc_auc", "pr_auc", "ks", "brier")
            }
            for seg in SEGMENTS
        },
        "feature_stats": _feature_stats(train_df, features),
        "train_rows": len(train_df),
        "base_default_rate": float(y_tr.mean()),
    }
    (settings.models_dir / META_FILE).write_text(
        json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    log(f"Saved {model_path} ({model_path.stat().st_size / 1e6:.1f} MB), version {version}")

    # --- synthetic sample for the demo / CI
    sample = make_synthetic_sample(train_df, input_features, CATEGORICAL, settings.seed)
    settings.sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = settings.sample_dir / "synthetic_applicants.csv"
    sample.to_csv(sample_path, index=False, float_format="%.6g", lineterminator="\n")
    log(f"Wrote {len(sample)} synthetic applicants -> {sample_path}")

    timings["total_s"] = time.perf_counter() - t_start
    results = {
        "run": {
            "date": trained_at,
            "model_version": version,
            "rows": len(df),
            "train_rows": len(train_df),
            "holdout_rows": len(holdout),
            "features": len(features),
            "seed": settings.seed,
            "folds": settings.n_folds,
            "holdout_fraction": settings.holdout_fraction,
            "timings_s": {k: round(v, 1) for k, v in timings.items()},
            "versions": {
                "python": platform.python_version(),
                "lightgbm": lgb.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "shap": explain.shap.__version__,
            },
            "platform": platform.platform(),
        },
        "data": {
            "default_rate": float(df["TARGET"].mean()),
            "thin_file_n": int(df["THIN_FILE"].sum()),
            "thin_file_default_rate": float(df.loc[df["THIN_FILE"] == 1, "TARGET"].mean()),
            "no_bureau_n": int(df["NO_BUREAU"].sum()),
            "no_bureau_default_rate": float(df.loc[df["NO_BUREAU"] == 1, "TARGET"].mean()),
            "holdout_thin_file_n": int(seg_ho["thin_file"].sum()),
            "holdout_thin_file_defaults": int(y_ho[seg_ho["thin_file"]].sum()),
        },
        "cost_assumptions": {
            **costs.as_dict(),
            "breakeven_probability": costs.breakeven_probability,
        },
        "imbalance": {
            "chosen": chosen,
            "negatives_per_positive": pos_ratio,
            "strategies": strategies,
        },
        "threshold": threshold,
        "num_boost_round": n_rounds,
        "holdout": evaluation,
        "oof_train": oof_segments,
        "segment_thresholds": segment_thresholds,
        "sensitivity": sensitivity,
        "fairness_audit": audit,
        "shap": {
            "sample_size": len(X_shap),
            "max_abs_diff_shap_vs_lightgbm_pred_contrib": shap_gap,
            "top_features": [
                {"feature": k, "mean_abs_shap": float(v)} for k, v in by_feature.head(20).items()
            ],
            "reason_groups": [{"group": k, "mean_abs_shap": float(v)} for k, v in by_group.items()],
        },
        "excluded_features": EXCLUDED_FEATURES,
        "audit_columns": AUDIT_COLUMNS,
    }
    write_reports(results, settings.reports_dir)
    log(f"Done in {timings['total_s'] / 60:.1f} min. Reports in {settings.reports_dir}")
    return results
