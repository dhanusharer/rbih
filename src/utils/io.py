from __future__ import annotations
import pandas as pd
import polars as pl
from pathlib import Path
from glob import glob
from typing import List
from src.utils.logger import get_logger

log = get_logger()


def find_parquet_files(directory: str | Path) -> List[Path]:
    files = sorted(Path(directory).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {directory}")
    log.info(f"Found {len(files)} parquet files in {directory}")
    return files


def read_static(path: str | Path, columns=None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_parquet(path, columns=columns)
    log.info(f"Loaded {path.name}: {df.shape}")
    return df


def save_features(df, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pl.DataFrame):
        df.write_parquet(str(path))
    else:
        df.to_parquet(path, index=False)
    log.info(f"Saved {path.name}: {len(df):,} rows → {path}")


def load_features(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    log.info(f"Loaded features {Path(path).name}: {df.shape}")
    return df
