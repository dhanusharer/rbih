"""
Phase 2 — Transaction Feature Engineering (Memory-Safe)
=========================================================
KEY FIXES vs v1:
  1. abs_amount split into separate with_columns call (Polars column ordering bug)
  2. transactions_additional has NO account_id — joins via transaction_id
  3. part_transaction_type (CI/BI/IP/IC) and transaction_sub_type (CLT_CASH/LOAN/NORMAL)
     now used as features per README schema
  4. time_windows uses mule_flag_date for accurate Temporal IoU
  5. streaming=True on all collect() calls for laptop RAM safety

Output: data/features/txn_features.parquet
        data/features/geo_features.parquet
        data/features/ip_features.parquet
        data/features/time_windows.parquet
"""
from __future__ import annotations
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm

from src.utils.config import CFG
from src.utils.logger import get_logger
from src.utils.io import save_features

log = get_logger()

STRUCT_LOW  = CFG.processing.structuring_lower
STRUCT_HIGH = CFG.processing.structuring_upper
ROUND_BINS  = CFG.processing.round_amount_bins
NIGHT_START = CFG.processing.night_hours_start
NIGHT_END   = CFG.processing.night_hours_end


def _round_amount_flag(amount: pl.Expr) -> pl.Expr:
    """1 if amount divisible by any round bin threshold."""
    result = (amount % ROUND_BINS[0] == 0)
    for b in ROUND_BINS[1:]:
        result = result | (amount % b == 0)
    return result.cast(pl.Int8)


def get_batch_files(batch_dir: str) -> list[str]:
    files = sorted(glob(f"{batch_dir}/**/*.parquet", recursive=True)) or \
            sorted(glob(f"{batch_dir}/*.parquet"))
    return files


# ─────────────────────────────────────────────────────────────────────────────
def build_txn_id_lookup(txn_batches: list[str]) -> pl.DataFrame:
    """
    Build transaction_id → account_id mapping from core transactions.
    Required because transactions_additional has NO account_id column.
    """
    log.info("Building transaction_id → account_id lookup...")
    frames = []
    for batch in tqdm(txn_batches, desc="ID lookup"):
        files = get_batch_files(batch)
        if not files:
            continue
        lf = pl.scan_parquet(files, low_memory=True)
        id_map = lf.select(["transaction_id", "account_id"]).collect(streaming=True)
        frames.append(id_map)
    result = pl.concat(frames).unique(subset=["transaction_id"])
    log.info(f"  Lookup built: {len(result):,} unique transaction_ids")
    return result


# ─────────────────────────────────────────────────────────────────────────────
def aggregate_core_txn(batch_dir: str) -> pl.DataFrame:
    """
    Aggregate one batch of core transactions to account level.
    FIX: abs_amount created in first with_columns, referenced in second.
    """
    files = get_batch_files(batch_dir)
    if not files:
        log.warning(f"No files in {batch_dir}")
        return pl.DataFrame()

    lf = pl.scan_parquet(files, low_memory=True)

    # Step 1: parse timestamp
    lf = lf.with_columns([
        pl.col("transaction_timestamp").cast(pl.Utf8)
          .str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False).alias("ts")
    ])

    # Step 2: create abs_amount and simple flags (NO cross-references)
    lf = lf.with_columns([
        pl.col("amount").abs().alias("abs_amount"),
        pl.col("ts").dt.hour().alias("hour"),
        (pl.col("amount") < 0).cast(pl.Int8).alias("is_reversal"),
    ])

    # Step 3: reference abs_amount and hour (now they exist)
    lf = lf.with_columns([
        (
            (pl.col("abs_amount") >= STRUCT_LOW) &
            (pl.col("abs_amount") < STRUCT_HIGH)
        ).cast(pl.Int8).alias("is_structuring"),
        (
            (pl.col("hour") >= NIGHT_START) &
            (pl.col("hour") < NIGHT_END)
        ).cast(pl.Int8).alias("is_night"),
        _round_amount_flag(pl.col("abs_amount")).alias("is_round"),
    ])

    agg = lf.group_by("account_id").agg([
        pl.len().alias("txn_count"),
        pl.col("abs_amount").sum().alias("total_volume"),
        pl.col("abs_amount").filter(pl.col("txn_type") == "C").sum().alias("total_credit"),
        pl.col("abs_amount").filter(pl.col("txn_type") == "D").sum().alias("total_debit"),
        pl.col("abs_amount").mean().alias("avg_amount"),
        pl.col("abs_amount").std().alias("std_amount"),
        pl.col("abs_amount").max().alias("max_amount"),

        pl.col("ts").min().alias("first_txn_ts"),
        pl.col("ts").max().alias("last_txn_ts"),
        pl.col("ts").dt.date().n_unique().alias("active_days"),

        pl.col("is_structuring").sum().alias("structuring_count"),
        pl.col("is_night").sum().alias("night_count"),
        pl.col("is_round").sum().alias("round_count"),
        pl.col("is_reversal").sum().alias("reversal_count"),

        pl.col("counterparty_id").n_unique().alias("unique_counterparties"),
        pl.col("counterparty_id").filter(pl.col("txn_type") == "C").n_unique().alias("unique_senders"),
        pl.col("counterparty_id").filter(pl.col("txn_type") == "D").n_unique().alias("unique_receivers"),
        pl.col("channel").n_unique().alias("channel_count"),

        # Month-boundary transactions (Pattern #11 — Salary Cycle Exploitation)
        pl.col("ts").filter(pl.col("ts").dt.day() <= 5).len().alias("month_start_txn_count"),

        # MCC diversity (Pattern #13 — MCC-Amount Anomaly)
        pl.col("mcc_code").n_unique().alias("unique_mcc_count"),
    ]).collect(streaming=True)

    return agg


# ─────────────────────────────────────────────────────────────────────────────
def aggregate_additional_txn(batch_dir: str, txn_id_to_account: pl.DataFrame) -> pl.DataFrame:
    """
    Process transactions_additional for geo + IP + balance features.
    FIX: No account_id in this file — must join via transaction_id.
    Uses part_transaction_type and transaction_sub_type per README schema.
    """
    files = get_batch_files(batch_dir)
    if not files:
        return pl.DataFrame()

    lf = pl.scan_parquet(files, low_memory=True)

    # JOIN to get account_id — this is the critical fix
    lf = lf.join(txn_id_to_account.lazy(), on="transaction_id", how="inner")

    # README: part_transaction_type = CI/BI/IP/IC
    # README: transaction_sub_type  = CLT_CASH/LOAN/NORMAL
    lf = lf.with_columns([
        (pl.col("part_transaction_type") == "CI").cast(pl.Int8).alias("is_customer_induced"),
        (pl.col("transaction_sub_type") == "CLT_CASH").cast(pl.Int8).alias("is_cash_txn"),
        (pl.col("transaction_sub_type") == "LOAN").cast(pl.Int8).alias("is_loan_txn"),
        (pl.col("balance_after_transaction").abs() < 100).cast(pl.Int8).alias("is_near_zero_bal"),
    ])

    agg = lf.group_by("account_id").agg([
        # IP features
        pl.col("ip_address").n_unique().alias("unique_ips"),
        pl.col("ip_address").value_counts().struct.field("count").max().alias("max_ip_reuse"),

        # Geo features
        pl.col("latitude").mean().alias("median_lat"),
        pl.col("longitude").mean().alias("median_lon"),
        pl.col("latitude").std().alias("lat_std"),
        pl.col("longitude").std().alias("lon_std"),

        # Balance features
        pl.col("balance_after_transaction").min().alias("min_balance_ever"),
        pl.col("balance_after_transaction").std().alias("balance_volatility"),
        pl.col("is_near_zero_bal").sum().alias("near_zero_balance_count"),

        # Transaction type features (new per README)
        pl.col("is_customer_induced").mean().alias("pct_customer_induced"),
        pl.col("is_cash_txn").mean().alias("pct_cash_txn"),
        pl.col("is_loan_txn").mean().alias("pct_loan_txn"),
    ]).collect(streaming=True)

    return agg


# ─────────────────────────────────────────────────────────────────────────────
def build_ip_shared_accounts(add_batches: list[str],
                              txn_id_to_account: pl.DataFrame) -> pl.DataFrame:
    """
    For each account: how many OTHER accounts share its IP?
    High value = red flag (Pattern #14 — IP Clustering).
    FIX: join via transaction_id to get account_id.
    """
    frames = []
    for d in add_batches:
        files = get_batch_files(d)
        if not files:
            continue
        lf = pl.scan_parquet(files, low_memory=True)
        lf = lf.join(txn_id_to_account.lazy(), on="transaction_id", how="inner")
        ip_counts = (
            lf.select(["ip_address", "account_id"])
            .unique()
            .group_by("ip_address")
            .agg(pl.col("account_id").n_unique().alias("ip_shared_accounts"))
            .collect(streaming=True)
        )
        frames.append(ip_counts)

    if not frames:
        return pl.DataFrame(schema={"account_id": pl.Utf8, "ip_shared_accounts_max": pl.Int64})

    combined_ip = pl.concat(frames).group_by("ip_address").agg(
        pl.col("ip_shared_accounts").max()
    )

    # Map back to account level
    acct_frames = []
    for d in add_batches:
        files = get_batch_files(d)
        if not files:
            continue
        lf = pl.scan_parquet(files, low_memory=True)
        lf = lf.join(txn_id_to_account.lazy(), on="transaction_id", how="inner")
        acct_ip = (
            lf.select(["ip_address", "account_id"]).unique()
            .join(combined_ip.lazy(), on="ip_address")
            .group_by("account_id")
            .agg(pl.col("ip_shared_accounts").max().alias("ip_shared_accounts_max"))
            .collect(streaming=True)
        )
        acct_frames.append(acct_ip)

    if not acct_frames:
        return pl.DataFrame(schema={"account_id": pl.Utf8, "ip_shared_accounts_max": pl.Int64})

    return pl.concat(acct_frames).group_by("account_id").agg(
        pl.col("ip_shared_accounts_max").max()
    )


# ─────────────────────────────────────────────────────────────────────────────
def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ratio and composite features from aggregated columns."""
    eps = 1e-9

    df["txn_span_days"] = (
        pd.to_datetime(df["last_txn_ts"], errors="coerce") -
        pd.to_datetime(df["first_txn_ts"], errors="coerce")
    ).dt.days.clip(lower=0).fillna(0)

    span = df["txn_span_days"].clip(lower=1)

    # Core ratios
    df["passthrough_ratio"]   = df["total_debit"] / (df["total_credit"] + eps)
    df["net_flow"]            = df["total_credit"] - df["total_debit"]
    df["structuring_score"]   = df["structuring_count"] / (df["txn_count"] + eps)
    df["night_ratio"]         = df["night_count"] / (df["txn_count"] + eps)
    df["round_amount_ratio"]  = df["round_count"] / (df["txn_count"] + eps)
    df["reversal_ratio"]      = df["reversal_count"] / (df["txn_count"] + eps)

    # Fan-in / fan-out (Pattern #4)
    df["fan_in_score"]        = df["unique_senders"] / (df["unique_receivers"] + eps)
    df["fan_out_score"]       = df["unique_receivers"] / (df["unique_senders"] + eps)
    df["cp_concentration"]    = 1.0 / (df["unique_counterparties"] + eps)

    # Velocity (Pattern #17)
    df["txn_rate_per_day"]    = df["txn_count"] / span
    df["volume_per_day"]      = df["total_volume"] / span
    df["activity_ratio"]      = df["active_days"] / (df["txn_span_days"] + 1)

    # Month-start ratio (Pattern #11 — Salary Cycle Exploitation)
    df["month_start_ratio"]   = df["month_start_txn_count"] / (df["txn_count"] + eps)

    # Channel diversity (Pattern #19)
    df["channel_diversity"]   = np.log1p(df["channel_count"])

    # Log transforms for skewed features
    for col in ["txn_count", "total_volume", "total_credit", "total_debit",
                "max_amount", "avg_amount", "unique_counterparties"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0).fillna(0))

    return df


# ─────────────────────────────────────────────────────────────────────────────
def compute_time_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate suspicious activity windows for Temporal IoU scoring.
    Strategy: suspicious_end = last_txn_ts,
              suspicious_start = max(first_txn_ts, last_txn_ts - 180 days)
    This captures the main activity burst before detection.
    mule_flag_date from train_labels will further refine these in merge step.
    """
    result = df[["account_id", "first_txn_ts", "last_txn_ts"]].copy()
    result["first_txn_ts"] = pd.to_datetime(result["first_txn_ts"], errors="coerce")
    result["last_txn_ts"]  = pd.to_datetime(result["last_txn_ts"],  errors="coerce")

    lookback = pd.Timedelta(days=180)
    computed_start = result["last_txn_ts"] - lookback
    result["suspicious_start"] = computed_start.where(
        computed_start > result["first_txn_ts"], result["first_txn_ts"]
    )
    result["suspicious_end"] = result["last_txn_ts"]

    return result[["account_id", "suspicious_start", "suspicious_end"]]


# ─────────────────────────────────────────────────────────────────────────────
def run() -> pd.DataFrame:
    log.info("=== Phase 2: Transaction Feature Engineering ===")

    txn_dir     = Path(CFG.paths.transactions)
    txn_add_dir = Path(CFG.paths.transactions_add)

    txn_batches = sorted([str(d) for d in txn_dir.iterdir() if d.is_dir()]) \
                  if txn_dir.exists() else [str(txn_dir)]
    add_batches = sorted([str(d) for d in txn_add_dir.iterdir() if d.is_dir()]) \
                  if txn_add_dir.exists() else [str(txn_add_dir)]

    # ── Step 1: Core transactions ─────────────────────────────────────────────
    log.info(f"Processing {len(txn_batches)} core transaction batch(es)...")
    core_frames = []
    for batch in tqdm(txn_batches, desc="Core txn batches"):
        frame = aggregate_core_txn(batch)
        if len(frame) > 0:
            core_frames.append(frame)

    if not core_frames:
        raise RuntimeError("No transaction data found. Check data/raw/transactions/")

    log.info("Merging core batch aggregates...")
    combined = pl.concat(core_frames)

    sum_cols = ["txn_count", "total_volume", "total_credit", "total_debit",
                "structuring_count", "night_count", "round_count", "reversal_count",
                "active_days", "month_start_txn_count"]
    max_cols = ["max_amount", "unique_counterparties", "unique_senders",
                "unique_receivers", "channel_count", "unique_mcc_count"]

    txn_feat = combined.group_by("account_id").agg(
        [pl.col(c).sum() for c in sum_cols if c in combined.columns] +
        [pl.col(c).max() for c in max_cols if c in combined.columns] +
        [pl.col("avg_amount").mean(),
         pl.col("std_amount").mean(),
         pl.col("first_txn_ts").min(),
         pl.col("last_txn_ts").max()]
    ).to_pandas()

    # ── CHECKPOINT: Save core features immediately so progress is not lost ────
    log.info("CHECKPOINT: Saving core transaction features...")
    save_features(txn_feat, CFG.paths.txn_features)
    log.info("Core features saved. Starting additional transactions...")

    # ── Step 2: Build transaction_id → account_id lookup ─────────────────────
    # Done batch-by-batch to avoid loading all 400M IDs into RAM at once
    txn_id_to_account = build_txn_id_lookup(txn_batches)

    # ── Step 3: Additional transactions (geo + IP + balance + txn types) ──────
    try:
        log.info(f"Processing {len(add_batches)} additional transaction batch(es)...")
        add_frames = []
        for batch in tqdm(add_batches, desc="Additional txn batches"):
            frame = aggregate_additional_txn(batch, txn_id_to_account)
            if len(frame) > 0:
                add_frames.append(frame)

        if add_frames:
            add_combined = pl.concat(add_frames).group_by("account_id").agg([
                pl.col("unique_ips").sum(),
                pl.col("max_ip_reuse").max(),
                pl.col("median_lat").mean(),
                pl.col("median_lon").mean(),
                pl.col("lat_std").max(),
                pl.col("lon_std").max(),
                pl.col("min_balance_ever").min(),
                pl.col("balance_volatility").mean(),
                pl.col("near_zero_balance_count").sum(),
                pl.col("pct_customer_induced").mean(),
                pl.col("pct_cash_txn").mean(),
                pl.col("pct_loan_txn").mean(),
            ]).to_pandas()

            geo_feat = add_combined[["account_id", "median_lat", "median_lon",
                                      "lat_std", "lon_std"]].copy()
            save_features(geo_feat, CFG.paths.geo_features)

            ip_feat = add_combined[["account_id", "unique_ips", "max_ip_reuse",
                                      "min_balance_ever", "balance_volatility",
                                      "near_zero_balance_count",
                                      "pct_customer_induced", "pct_cash_txn",
                                      "pct_loan_txn"]].copy()
            save_features(ip_feat, CFG.paths.ip_features)

            txn_feat = txn_feat.merge(add_combined, on="account_id", how="left")
            log.info("Additional transaction features merged successfully.")
        else:
            log.warning("No additional frames produced — skipping geo/IP features.")

    except Exception as e:
        log.warning(f"Additional transactions failed (non-fatal): {e}")
        log.warning("Core features already saved — continuing without geo/IP features.")

    # ── Step 4: IP sharing across accounts ────────────────────────────────────
    log.info("Computing cross-account IP sharing features...")
    try:
        ip_shared = build_ip_shared_accounts(add_batches, txn_id_to_account).to_pandas()
        txn_feat = txn_feat.merge(ip_shared, on="account_id", how="left")
    except Exception as e:
        log.warning(f"IP sharing failed (non-fatal): {e}")
        txn_feat["ip_shared_accounts_max"] = 0

    # ── Step 5: Derived features ──────────────────────────────────────────────
    log.info("Computing derived ratio features...")
    txn_feat = compute_derived_features(txn_feat)

    # ── Step 6: Time windows for Temporal IoU ─────────────────────────────────
    log.info("Computing temporal activity windows...")
    time_wins = compute_time_windows(txn_feat)
    save_features(time_wins, CFG.paths.time_windows)

    # Final save overwrites checkpoint with fully enriched version
    save_features(txn_feat, CFG.paths.txn_features)
    log.info(f"Transaction features complete: {txn_feat.shape}")
    return txn_feat


if __name__ == "__main__":
    run()