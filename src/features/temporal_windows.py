"""
Phase 22 — Precise Temporal Window Detection
Data-driven parameters from analyze_windows.py:
  - Median first_txn → flag_date: 678 days
  - Median last_txn  → flag_date: -13 days (activity continues after flag)
  - Median span: 757 days
  - flag_position median: 1.0 (flag = end of activity)
"""
from __future__ import annotations
from glob import glob
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger()

MEDIAN_EXTENSION_DAYS = 14
MEDIAN_LOOKBACK_DAYS  = 678
MEDIAN_SPAN_DAYS      = 757


def build_temporal_windows(txn_batches: list[str]) -> pd.DataFrame:
    """Compute first/last txn per account by chunked file reading — RAM safe."""
    log.info("Building temporal windows (chunked file scan)...")

    all_files = []
    for batch in txn_batches:
        all_files += sorted(glob(f"{batch}/**/*.parquet", recursive=True)) or \
                     sorted(glob(f"{batch}/*.parquet"))

    if not all_files:
        log.warning("No transaction files found")
        return pd.DataFrame()

    log.info(f"  Scanning {len(all_files)} files...")

    # Running min/max per account — never load more than one file at a time
    bounds = {}  # account_id -> [first_txn, last_txn]

    for f in tqdm(all_files, desc="Window scan"):
        try:
            # transactions_additional has no account_id — skip it
            import pyarrow.parquet as pq
            schema_names = pq.read_schema(f).names
            if "account_id" not in schema_names:
                continue
            df = pd.read_parquet(f, columns=["account_id", "transaction_timestamp"])
            df["ts"] = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
            df = df.dropna(subset=["ts"])

            for acc, grp in df.groupby("account_id")["ts"]:
                mn, mx = grp.min(), grp.max()
                if acc not in bounds:
                    bounds[acc] = [mn, mx]
                else:
                    if mn < bounds[acc][0]: bounds[acc][0] = mn
                    if mx > bounds[acc][1]: bounds[acc][1] = mx
        except Exception as e:
            log.warning(f"  Skipped {f}: {e}")
            continue

    log.info(f"  Got bounds for {len(bounds):,} accounts")

    result = pd.DataFrame([
        {"account_id": acc, "first_txn": v[0], "last_txn": v[1]}
        for acc, v in bounds.items()
    ])

    # Load labels for anchoring
    labels_path = Path(CFG.paths.train_labels)
    if labels_path.exists():
        labels = pd.read_parquet(labels_path)
        mules = labels[labels["is_mule"] == 1][["account_id", "mule_flag_date"]].copy()
        mules["mule_flag_date"] = pd.to_datetime(mules["mule_flag_date"], errors="coerce")
        result = result.merge(mules, on="account_id", how="left")
    else:
        result["mule_flag_date"] = pd.NaT

    has_flag = result["mule_flag_date"].notna()
    log.info(f"  Known mules with flag_date: {has_flag.sum():,}")

    # Default: use actual transaction bounds
    result["suspicious_start"] = result["first_txn"]
    result["suspicious_end"]   = result["last_txn"]

    # For known mules: end = flag_date + 14 days (activity continues slightly after flag)
    result.loc[has_flag, "suspicious_end"] = (
        result.loc[has_flag, "mule_flag_date"] + pd.Timedelta(days=MEDIAN_EXTENSION_DAYS)
    )
    # Cap at last_txn + 30 days
    cap = result.loc[has_flag, "last_txn"] + pd.Timedelta(days=30)
    result.loc[has_flag, "suspicious_end"] = result.loc[has_flag, "suspicious_end"].where(
        result.loc[has_flag, "suspicious_end"] <= cap, cap
    )

    # Cap start at flag_date - (678 + 180) days to avoid over-wide windows
    max_lookback = pd.Timedelta(days=MEDIAN_LOOKBACK_DAYS + 180)
    early_start = result.loc[has_flag, "mule_flag_date"] - max_lookback
    result.loc[has_flag, "suspicious_start"] = result.loc[has_flag, "suspicious_start"].where(
        result.loc[has_flag, "suspicious_start"] >= early_start, early_start
    )

    # Fix any start >= end
    bad = result["suspicious_start"] >= result["suspicious_end"]
    if bad.sum() > 0:
        result.loc[bad, "suspicious_start"] = (
            result.loc[bad, "suspicious_end"] - pd.Timedelta(days=MEDIAN_SPAN_DAYS)
        )

    result = result[["account_id", "suspicious_start", "suspicious_end"]]

    # ── Fill missing accounts from accounts.parquet ───────────────────────────
    accounts_path = Path(CFG.paths.accounts)
    if accounts_path.exists():
        accts = pd.read_parquet(accounts_path, columns=["account_id", "account_opening_date"])
        accts["account_opening_date"] = pd.to_datetime(accts["account_opening_date"], errors="coerce")
        missing = accts[~accts["account_id"].isin(result["account_id"])].copy()
        if len(missing) > 0:
            # Fallback: open_date → open_date + 757 days (median mule span)
            missing["suspicious_start"] = missing["account_opening_date"]
            missing["suspicious_end"]   = missing["account_opening_date"] + pd.Timedelta(days=MEDIAN_SPAN_DAYS)
            missing = missing[["account_id", "suspicious_start", "suspicious_end"]]
            result = pd.concat([result, missing], ignore_index=True)
            log.info(f"  Added fallback windows for {len(missing):,} accounts from accounts.parquet")

    span = (result["suspicious_end"] - result["suspicious_start"]).dt.days
    log.info(f"  Total accounts: {len(result):,} | Median span: {span.median():.0f} days")

    return result


def run():
    log.info("=== Phase 22: Precise Temporal Window Detection ===")

    # Scan BOTH transactions and transactions_additional
    txn_dir = CFG.paths.transactions
    txn_add = CFG.paths.transactions_add
    batches = (sorted(glob(f"{txn_dir}/batch-*")) or [txn_dir]) +               (sorted(glob(f"{txn_add}/batch-*")) or [txn_add])
    log.info(f"Scanning {len(batches)} batch folders")

    df = build_temporal_windows(batches)
    if len(df) == 0:
        log.error("No windows computed")
        return

    out_path = CFG.paths.time_windows
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(out_path).exists():
        Path(out_path).unlink()

    df.to_parquet(out_path, index=False)
    log.info(f"Saved: {len(df):,} windows → {out_path}")


if __name__ == "__main__":
    run()