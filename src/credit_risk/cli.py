"""Command-line entry point: `uv run credit-risk --help`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from credit_risk.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


@app.callback()
def _main() -> None:
    # .env is read from the working directory only (never searched for up the tree).
    load_dotenv(Path.cwd() / ".env", override=False)


@app.command()
def download(
    force: bool = typer.Option(False, help="Re-download files that already exist."),
) -> None:
    """Download the 7 Home Credit tables from Kaggle (needs KAGGLE_API_TOKEN) and convert to Parquet."""
    from credit_risk.data.download import download as do_download
    from credit_risk.data.download import to_parquet

    settings = get_settings()
    print(f"Downloading to {settings.raw_dir}")
    do_download(settings, force=force)
    print("Converting to Parquet")
    to_parquet(settings, force=force)
    print("Done.")


@app.command()
def features() -> None:
    """Build data/processed/features.parquet from the downloaded tables."""
    from credit_risk.features.build import build_and_save

    build_and_save(get_settings())


@app.command()
def tune(trials: int = typer.Option(40, help="Optuna trial budget.")) -> None:
    """Search LightGBM hyper-parameters on the training split; writes config/lgbm_params.json."""
    import pandas as pd

    from credit_risk.features.application import CATEGORICAL
    from credit_risk.features.build import build_and_save, feature_columns
    from credit_risk.modeling.split import holdout_split
    from credit_risk.modeling.train import PARAMS_PATH
    from credit_risk.modeling.tune import tune as run_tune
    from credit_risk.scoring.frame import prepare_frame

    settings = get_settings()
    if settings.features_path.exists():
        df = pd.read_parquet(settings.features_path)
    else:
        df = build_and_save(settings)
    train_df, _ = holdout_split(df, settings.holdout_fraction, settings.seed)
    categories = {c: sorted(train_df[c].dropna().unique().tolist()) for c in CATEGORICAL}
    X = prepare_frame(train_df, feature_columns(df), categories)
    run_tune(X, train_df["TARGET"].to_numpy(), trials, settings.seed, PARAMS_PATH)


@app.command()
def train(
    rebuild_features: bool = typer.Option(False, help="Rebuild features even if cached."),
) -> None:
    """Train, evaluate and explain; writes models/, reports/, docs/img/ and the synthetic sample."""
    from credit_risk.modeling.train import train as run_train

    run_train(get_settings(), rebuild_features=rebuild_features)


@app.command()
def sample() -> None:
    """Regenerate only data/sample/synthetic_applicants.csv (same code and seed as `train`)."""
    import pandas as pd

    from credit_risk.features.application import CATEGORICAL
    from credit_risk.features.build import input_feature_columns
    from credit_risk.modeling.split import holdout_split
    from credit_risk.synth import write_sample

    settings = get_settings()
    df = pd.read_parquet(settings.features_path)
    train_df, _ = holdout_split(df, settings.holdout_fraction, settings.seed)
    path = settings.sample_dir / "synthetic_applicants.csv"
    out = write_sample(train_df, input_feature_columns(df), CATEGORICAL, settings.seed, path)
    print(f"Wrote {len(out)} synthetic applicants (0 copies of real rows) -> {path}")


@app.command("compare-metrics")
def compare_metrics(
    actual: Annotated[Path, typer.Argument(help="metrics.json from a fresh run.")] = Path(
        "reports/metrics.json"
    ),
    expected: Annotated[
        Path | None,
        typer.Option(help="Reference metrics.json. Default: the committed version (git HEAD)."),
    ] = None,
    tol: Annotated[float, typer.Option(help="Allowed absolute difference per metric.")] = 0.005,
) -> None:
    """Check that a re-run reproduces the committed hold-out metrics within `tol`."""
    import json
    import subprocess

    from credit_risk.modeling.compare import compare

    if expected is None:
        committed = subprocess.run(
            ["git", "show", "HEAD:reports/metrics.json"], capture_output=True, text=True, check=True
        ).stdout
        reference = json.loads(committed)
        print("Reference: reports/metrics.json at git HEAD")
    else:
        reference = json.loads(expected.read_text(encoding="utf-8"))
    rows, ok = compare(reference, json.loads(actual.read_text(encoding="utf-8")), tol)
    print(f"{'':5}{'metric':<55}{'reference':>10}{'this run':>10}{'diff':>9}")
    for name, a, b, diff, passed in rows:
        print(f"{'ok  ' if passed else 'FAIL'} {name:<55} {a:>9.4f} {b:>9.4f} {diff:+.4f}")
    print("All within tolerance." if ok else "Some metrics differ by more than the tolerance.")
    raise typer.Exit(0 if ok else 1)


if __name__ == "__main__":
    app()
