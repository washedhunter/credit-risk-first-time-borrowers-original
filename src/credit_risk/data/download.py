"""Download the Home Credit tables from Kaggle and convert them to Parquet."""

from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path

from credit_risk.config import COMPETITION, RAW_TABLES, Settings


def _kaggle_api():
    if not (os.environ.get("KAGGLE_API_TOKEN") or (Path.home() / ".kaggle").exists()):
        raise RuntimeError(
            "No Kaggle credentials. Put KAGGLE_API_TOKEN in .env (see .env.example) and accept the "
            f"competition rules at https://www.kaggle.com/competitions/{COMPETITION}/rules"
        )
    # Imported lazily: `import kaggle` tries to authenticate as a side effect.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def download(settings: Settings, force: bool = False) -> list[Path]:
    """Fetch each table's CSV into data/raw/ (skips files already there)."""
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    api = None
    paths = []
    for table in RAW_TABLES:
        csv_path = settings.raw_dir / f"{table}.csv"
        paths.append(csv_path)
        if csv_path.exists() and not force:
            print(f"  have {csv_path.name}")
            continue
        api = api or _kaggle_api()
        started = time.perf_counter()
        api.competition_download_file(
            COMPETITION, f"{table}.csv", path=str(settings.raw_dir), force=True, quiet=True
        )
        _unzip_if_needed(settings.raw_dir, table)
        size_mb = csv_path.stat().st_size / 1e6
        print(f"  got  {csv_path.name} ({size_mb:,.0f} MB, {time.perf_counter() - started:.0f}s)")
    return paths


def _unzip_if_needed(raw_dir: Path, table: str) -> None:
    for candidate in (raw_dir / f"{table}.csv.zip", raw_dir / f"{table}.zip"):
        if candidate.exists():
            with zipfile.ZipFile(candidate) as zf:
                zf.extractall(raw_dir)
            candidate.unlink()


def to_parquet(settings: Settings, force: bool = False) -> None:
    """CSV -> Parquet once, so feature builds read in seconds instead of minutes."""
    import pandas as pd

    settings.interim_dir.mkdir(parents=True, exist_ok=True)
    for table in RAW_TABLES:
        out = settings.interim_dir / f"{table}.parquet"
        if out.exists() and not force:
            continue
        df = pd.read_csv(settings.raw_dir / f"{table}.csv")
        # Floats stay float64 so training sees exactly the values an API client would send;
        # only integer columns are downcast.
        for col in df.select_dtypes("int64").columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        df.to_parquet(out, index=False)
        print(f"  {table}: {len(df):,} rows x {df.shape[1]} cols -> {out.name}")
