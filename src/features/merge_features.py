"""
Phase 3 — Feature Merger
Combines static + transaction features.
Enriches time windows using mule_flag_date from train_labels.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import CFG
from src.utils.logger import get_logger
from src.utils.io import read_static, load_features, save_features

log = get_logger()


def enrich_time_windows(time_windows_path: str, labels_path: str) -> None:
    """
    Use mule_flag_date from train_labels as ground-truth suspicious_end.
    This directly improves Temporal IoU score.
    """
    tw = pd.read_parquet(time_windows_path)
    labels = pd.read_parquet(labels_path)

    mules = labels[labels["is_mule"] == 1][["account_id", "mule_flag_date"]].copy()
    mules["mule_flag_date"] = pd.to_datetime(mules["mule_flag_date"], errors="coerce")

    tw = tw.merge(mules, on="account_id", how="left")

    has_flag = tw["mule_flag_date"].notna()
    # For flagged mules: end = flag date, start = flag date - 180 days
    tw.loc[has_flag, "suspicious_end"]   = tw.loc[has_flag, "mule_flag_date"]
    tw.loc[has_flag, "suspicious_start"] = (
        tw.loc[has_flag, "mule_flag_date"] - pd.Timedelta(days=180)
    )
    tw.drop(columns=["mule_flag_date"], inplace=True)
    tw.to_parquet(time_windows_path, index=False)
    log.info(f"Time windows enriched with mule_flag_date for {has_flag.sum()} mule accounts")


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("=== Phase 3: Merging All Features ===")

    static  = load_features(CFG.paths.static_features)
    txn     = load_features(CFG.paths.txn_features)
    labels  = read_static(CFG.paths.train_labels)
    test_df = read_static(CFG.paths.test_accounts)

    # Merge static + transaction features
    df = static.merge(txn, on="account_id", how="left")

    # Merge graph features if available
    graph_path = Path(CFG.paths.features) / "graph_features.parquet"
    if graph_path.exists():
        graph = load_features(str(graph_path))
        df = df.merge(graph, on="account_id", how="left")
        log.info(f"Graph features merged: {graph.shape[1]-1} additional features")
    else:
        log.warning("Graph features not found — run phase 2b first for best results")

    # Fill NaN (accounts with no transactions get 0)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Drop any remaining non-numeric columns except account_id
    obj_cols = [c for c in df.select_dtypes(include=["object", "datetime64[ns]"]).columns
                if c != "account_id"]
    df.drop(columns=obj_cols, inplace=True, errors="ignore")

    # Enrich time windows with mule_flag_date
    tw_path = Path(CFG.paths.time_windows)
    if tw_path.exists():
        enrich_time_windows(str(tw_path), CFG.paths.train_labels)

        # Also ensure test accounts have time windows from txn features
        tw = pd.read_parquet(tw_path)
        test_missing = df_test[~df_test["account_id"].isin(tw["account_id"])]
        if len(test_missing) > 0 and "first_txn_ts" in df.columns:
            extra_tw = df[df["account_id"].isin(test_missing["account_id"])][
                ["account_id", "first_txn_ts", "last_txn_ts"]].copy()
            extra_tw["first_txn_ts"] = pd.to_datetime(extra_tw["first_txn_ts"], errors="coerce")
            extra_tw["last_txn_ts"]  = pd.to_datetime(extra_tw["last_txn_ts"],  errors="coerce")
            lookback = pd.Timedelta(days=180)
            extra_tw["suspicious_start"] = (extra_tw["last_txn_ts"] - lookback).where(
                extra_tw["last_txn_ts"] - lookback > extra_tw["first_txn_ts"],
                extra_tw["first_txn_ts"])
            extra_tw["suspicious_end"] = extra_tw["last_txn_ts"]
            extra_tw = extra_tw[["account_id", "suspicious_start", "suspicious_end"]]
            tw_combined = pd.concat([tw, extra_tw], ignore_index=True)
            tw_combined.to_parquet(tw_path, index=False)
            log.info(f"Added time windows for {len(extra_tw)} additional test accounts")

    # Split train / test
    df_train = df[df["account_id"].isin(labels["account_id"])].copy()
    df_train = df_train.merge(labels[["account_id", "is_mule"]], on="account_id", how="left")

    df_test = df[df["account_id"].isin(test_df["account_id"])].copy()

    log.info(f"Train: {df_train.shape} | Mule rate: {df_train['is_mule'].mean():.2%}")
    log.info(f"Test:  {df_test.shape}")

    save_features(df_train, CFG.paths.full_features)
    save_features(df_test,  CFG.paths.full_features_test)

    return df_train, df_test


if __name__ == "__main__":
    run()