"""
Phase 22 — Precise Temporal Window Detection
=============================================
From data analysis on 2,683 mules:
  - Median account span:          757 days
  - Median flag_date position:    end of activity (flag_position ~1.0)
  - Median first_txn → flag:      678 days before flag
  - Median last_txn  → flag:      -13 days (activity continues 13 days AFTER flag)

Therefore:
  suspicious_start = first_txn date  (or flag_date - 678 days as fallback)
  suspicious_end   = last_txn date   (or flag_date + 14 days as fallback)

For train mules with known flag_date:
  - Use actual first_txn and last_txn from transaction data (most precise)
  - Anchor to flag_date ± buffer only when no txn data available
"""
from __future__ import annotations
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger()

# From data analysis
MEDIAN_LOOKBACK_DAYS   = 678   # days before flag_date that mule activity starts
MEDIAN_EXTENSION_DAYS  = 14    # days after flag_date that activity continues
MEDIAN_SPAN_DAYS       = 757   # typical mule active period


def build_temporal_windows(txn_batches: list[str]) -> pd.DataFrame:
    """
    Compute precise suspicious_start and suspicious_end per account
    using actual first and last transaction dates.
    
    Strategy (in order of precision):
    1. Use actual first_txn and last_txn dates from transaction data
    2. For accounts with flag_date: anchor end to flag_date + 14 days
    3. Fallback: flag_date - 678 days → flag_date + 14 days
    """
    log.info("Building temporal windows from transaction data...")

    all_files = []
    for batch in txn_batches:
        all_files += sorted(glob(f"{batch}/**/*.parquet", recursive=True)) or \
                     sorted(glob(f"{batch}/*.parquet"))
    if not all_files:
        log.warning("No transaction files found")
        return pd.DataFrame()

    lf = pl.scan_parquet(all_files, low_memory=True)

    log.info("  Computing first/last transaction dates per account...")
    txn_bounds = (
        lf.with_columns(
            pl.col("transaction_timestamp").cast(pl.Utf8)
              .str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False).alias("ts")
        )
        .group_by("account_id")
        .agg([
            pl.col("ts").min().alias("first_txn"),
            pl.col("ts").max().alias("last_txn"),
            pl.len().alias("txn_count"),
            # Weekly volume std — high std = bursty = mule-like
            pl.col("amount").abs().std().alias("amount_std"),
        ])
        .collect(streaming=True)
    ).to_pandas()

    txn_bounds["first_txn"] = pd.to_datetime(txn_bounds["first_txn"])
    txn_bounds["last_txn"]  = pd.to_datetime(txn_bounds["last_txn"])

    log.info(f"  Got txn bounds for {len(txn_bounds):,} accounts")

    # Load labels for anchoring
    labels_path = Path(CFG.paths.train_labels)
    if labels_path.exists():
        labels = pd.read_parquet(labels_path)
        mules = labels[labels["is_mule"] == 1][["account_id", "mule_flag_date"]].copy()
        mules["mule_flag_date"] = pd.to_datetime(mules["mule_flag_date"], errors="coerce")
        txn_bounds = txn_bounds.merge(mules, on="account_id", how="left")
    else:
        txn_bounds["mule_flag_date"] = pd.NaT

    has_flag = txn_bounds["mule_flag_date"].notna()
    log.info(f"  Known mules with flag_date: {has_flag.sum():,}")

    # ── Compute windows ───────────────────────────────────────────────────────

    # Default: use actual transaction bounds
    txn_bounds["suspicious_start"] = txn_bounds["first_txn"]
    txn_bounds["suspicious_end"]   = txn_bounds["last_txn"]

    # For known mules: anchor end to flag_date + buffer
    # Data shows activity continues ~14 days after flag
    txn_bounds.loc[has_flag, "suspicious_end"] = (
        txn_bounds.loc[has_flag, "mule_flag_date"] + pd.Timedelta(days=MEDIAN_EXTENSION_DAYS)
    )

    # But don't extend past actual last_txn + 30 days
    cap = txn_bounds.loc[has_flag, "last_txn"] + pd.Timedelta(days=30)
    txn_bounds.loc[has_flag, "suspicious_end"] = txn_bounds.loc[has_flag, "suspicious_end"].where(
        txn_bounds.loc[has_flag, "suspicious_end"] <= cap,
        cap
    )

    # Start = actual first_txn (already set above)
    # But for known mules: if first_txn is very old (>3 years before flag),
    # cap it at flag_date - MEDIAN_LOOKBACK_DAYS to avoid over-wide windows
    max_lookback = pd.Timedelta(days=MEDIAN_LOOKBACK_DAYS + 180)  # +6mo buffer
    early_start = txn_bounds.loc[has_flag, "mule_flag_date"] - max_lookback
    txn_bounds.loc[has_flag, "suspicious_start"] = txn_bounds.loc[has_flag, "suspicious_start"].where(
        txn_bounds.loc[has_flag, "suspicious_start"] >= early_start,
        early_start
    )

    # For test accounts (no flag_date): use full txn range
    # This is actually correct — test mules will have activity for ~757 days
    # so first_txn → last_txn is a good estimate

    result = txn_bounds[["account_id", "suspicious_start", "suspicious_end"]].copy()

    # Sanity check: ensure start < end
    bad = result["suspicious_start"] >= result["suspicious_end"]
    if bad.sum() > 0:
        log.warning(f"  Fixing {bad.sum()} windows where start >= end")
        result.loc[bad, "suspicious_start"] = (
            result.loc[bad, "suspicious_end"] - pd.Timedelta(days=MEDIAN_SPAN_DAYS)
        )

    log.info(f"  Final windows: {len(result):,} accounts")

    span = (result["suspicious_end"] - result["suspicious_start"]).dt.days
    log.info(f"  Median span: {span.median():.0f} days | Mean: {span.mean():.0f} days")

    return result


def run():
    log.info("=== Phase 22: Precise Temporal Window Detection ===")

    txn_dir = CFG.paths.transactions
    batches = sorted(glob(f"{txn_dir}/batch-*")) or [txn_dir]

    df = build_temporal_windows(batches)

    if len(df) == 0:
        log.error("No windows computed — check transaction paths")
        return

    out_path = CFG.paths.time_windows
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Delete old windows file so we start fresh
    if Path(out_path).exists():
        Path(out_path).unlink()
        log.info("  Deleted old time_windows.parquet")

    df.to_parquet(out_path, index=False)
    log.info(f"Saved: {len(df):,} windows → {out_path}")


if __name__ == "__main__":
    run()