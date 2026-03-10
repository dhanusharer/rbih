"""
Phase 1 — Static Feature Engineering
Builds features from all small files. Fast (~5 min).
Schema verified against README v2.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from src.utils.config import CFG
from src.utils.logger import get_logger
from src.utils.io import read_static, save_features

log = get_logger()
REFERENCE_DATE = pd.Timestamp("2025-07-01")


def build_account_features(df_acc: pd.DataFrame, df_train: pd.DataFrame | None) -> pd.DataFrame:
    df = df_acc.copy()

    # README confirmed columns: account_opening_date, last_mobile_update_date,
    # last_kyc_date, freeze_date, unfreeze_date, branch_pin (NOT branch_pin_code)
    for col in ["account_opening_date", "last_mobile_update_date",
                "last_kyc_date", "freeze_date", "unfreeze_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["account_age_days"]          = (REFERENCE_DATE - df["account_opening_date"]).dt.days.clip(lower=0)
    df["days_since_mobile_update"]  = (REFERENCE_DATE - df["last_mobile_update_date"]).dt.days.clip(lower=0)
    df["days_since_kyc"]            = (REFERENCE_DATE - df["last_kyc_date"]).dt.days.clip(lower=0)
    df["is_new_account"]            = (df["account_age_days"] < CFG.processing.new_account_threshold_days).astype(int)
    df["freeze_ever"]               = df["freeze_date"].notna().astype(int)
    df["was_unfrozen"]              = df["unfreeze_date"].notna().astype(int)
    df["freeze_unfreeze_both"]      = (df["freeze_ever"] & df["was_unfrozen"]).astype(int)
    df["mobile_update_recent"]      = (df["days_since_mobile_update"] < 30).astype(int)
    df["kyc_stale"]                 = (df["days_since_kyc"] > 730).astype(int)

    df["product_family_enc"]  = df["product_family"].map({"S": 0, "K": 1, "O": 2}).fillna(-1).astype(int)
    df["kyc_compliant_enc"]   = (df["kyc_compliant"].fillna("N") == "Y").astype(int)
    df["nomination_flag_enc"] = (df["nomination_flag"].fillna("N") == "Y").astype(int)
    df["rural_branch_enc"]    = (df["rural_branch"].fillna("N") == "Y").astype(int)
    df["is_frozen_now"]       = (df["account_status"] == "frozen").astype(int)
    df["cheque_availed_enc"]  = (df["cheque_availed"].fillna("N") == "Y").astype(int)

    df["avg_balance_log"]          = np.log1p(df["avg_balance"].clip(lower=0).fillna(0))
    df["balance_negative"]         = (df["avg_balance"].fillna(0) < 0).astype(int)
    df["monthly_avg_balance_log"]  = np.log1p(df["monthly_avg_balance"].clip(lower=0).fillna(0))
    df["daily_avg_balance_log"]    = np.log1p(df["daily_avg_balance"].clip(lower=0).fillna(0))
    df["num_chequebooks"]          = df["num_chequebooks"].fillna(0).astype(int)

    # Branch mule density from training labels (Pattern #12 — Branch Collusion)
    if df_train is not None and "flagged_by_branch" in df_train.columns:
        branch_mule = (
            df_train[df_train["is_mule"] == 1]["flagged_by_branch"]
            .value_counts().rename("branch_mule_count")
        )
        df = df.merge(branch_mule, left_on="branch_code", right_index=True, how="left")
        df["branch_mule_count"] = df["branch_mule_count"].fillna(0)
    else:
        df["branch_mule_count"] = 0

    keep = [
        "account_id", "branch_code",
        "account_age_days", "days_since_mobile_update", "days_since_kyc",
        "is_new_account", "freeze_ever", "was_unfrozen", "freeze_unfreeze_both",
        "mobile_update_recent", "kyc_stale", "product_family_enc",
        "kyc_compliant_enc", "nomination_flag_enc", "rural_branch_enc",
        "is_frozen_now", "cheque_availed_enc", "avg_balance_log",
        "balance_negative", "monthly_avg_balance_log", "daily_avg_balance_log",
        "num_chequebooks", "branch_mule_count",
    ]
    return df[[c for c in keep if c in df.columns]]


def build_customer_features(df_cust: pd.DataFrame) -> pd.DataFrame:
    df = df_cust.copy()
    # README columns confirmed: date_of_birth, relationship_start_date,
    # pan_available, aadhaar_available, passport_available, mobile_banking_flag,
    # internet_banking_flag, atm_card_flag, demat_flag, credit_card_flag,
    # fastag_flag, customer_pin, permanent_pin
    df["dob"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["rel_start"] = pd.to_datetime(df["relationship_start_date"], errors="coerce")
    df["customer_age_years"]   = ((REFERENCE_DATE - df["dob"]).dt.days / 365.25).clip(lower=0)
    df["customer_tenure_days"] = (REFERENCE_DATE - df["rel_start"]).dt.days.clip(lower=0)

    flag_cols = ["pan_available", "aadhaar_available", "passport_available",
                 "mobile_banking_flag", "internet_banking_flag", "atm_card_flag",
                 "demat_flag", "credit_card_flag", "fastag_flag"]
    for c in flag_cols:
        df[f"{c}_enc"] = (df[c].fillna("N") == "Y").astype(int)

    # KYC weakness score: missing key docs + limited digital access
    df["kyc_weakness_score"] = (
        (1 - df["pan_available_enc"]) +
        (1 - df["aadhaar_available_enc"]) +
        (1 - df["mobile_banking_flag_enc"]) +
        (1 - df["internet_banking_flag_enc"])
    )
    df["digital_banking_score"] = (
        df["mobile_banking_flag_enc"] +
        df["internet_banking_flag_enc"] +
        df["atm_card_flag_enc"]
    )
    # PIN mismatch: residential vs permanent address differ
    df["pin_mismatch"] = (
        df["customer_pin"].astype(str) != df["permanent_pin"].astype(str)
    ).astype(int)

    keep = ["customer_id", "customer_age_years", "customer_tenure_days",
            "kyc_weakness_score", "digital_banking_score", "pin_mismatch"] + \
           [f"{c}_enc" for c in flag_cols]
    return df[[c for c in keep if c in df.columns]]


def build_product_features(df_prod: pd.DataFrame) -> pd.DataFrame:
    df = df_prod.copy()
    # README columns: loan_sum, loan_count, cc_sum, cc_count, od_sum, od_count,
    # ka_sum, ka_count, sa_sum, sa_count
    df["product_complexity"] = (
        df["loan_count"].fillna(0) + df["cc_count"].fillna(0) +
        df["od_count"].fillna(0) + df["ka_count"].fillna(0)
    )
    df["has_loan"]        = (df["loan_count"].fillna(0) > 0).astype(int)
    df["has_credit_card"] = (df["cc_count"].fillna(0) > 0).astype(int)
    df["has_overdraft"]   = (df["od_count"].fillna(0) > 0).astype(int)
    df["total_balance"]   = df["sa_sum"].fillna(0) + df["ka_sum"].fillna(0)
    df["total_balance_log"] = np.log1p(df["total_balance"].clip(lower=0))
    df["total_debt"]      = df["loan_sum"].fillna(0) + df["cc_sum"].fillna(0)
    df["debt_to_balance"] = df["total_debt"].abs() / (df["total_balance"].abs() + 1)
    return df[["customer_id", "product_complexity", "has_loan",
               "has_credit_card", "has_overdraft",
               "total_balance_log", "debt_to_balance"]]


def build_demographics_features(df_demo: pd.DataFrame) -> pd.DataFrame:
    df = df_demo.copy()
    # README columns: customer_id, name, gender, address_last_update_date,
    # address, phone_number, passbook_last_update_date,
    # joint_account_flag, nri_flag
    df["addr_update"] = pd.to_datetime(df["address_last_update_date"], errors="coerce")
    df["passbook_update"] = pd.to_datetime(df["passbook_last_update_date"], errors="coerce")
    df["address_freshness_days"]   = (REFERENCE_DATE - df["addr_update"]).dt.days.clip(lower=0)
    df["passbook_staleness_days"]  = (REFERENCE_DATE - df["passbook_update"]).dt.days.clip(lower=0)

    # These ARE in demographics per README
    df["joint_account_enc"] = (df["joint_account_flag"].fillna("N") == "Y").astype(int) \
        if "joint_account_flag" in df.columns else 0
    df["nri_flag_enc"] = (df["nri_flag"].fillna("N") == "Y").astype(int) \
        if "nri_flag" in df.columns else 0
    df["gender_enc"] = (df["gender"].fillna("M") == "M").astype(int) \
        if "gender" in df.columns else 0

    return df[["customer_id", "address_freshness_days", "passbook_staleness_days",
               "joint_account_enc", "nri_flag_enc", "gender_enc"]]


def build_scheme_features(df_scheme: pd.DataFrame) -> pd.DataFrame:
    df = df_scheme.copy()
    # README: scheme_code values: PMJDY, PMSBY, PMJJBY, APY, SCSS, SSA, REGULAR
    # PMJDY accounts are heavily targeted for mule activity
    df["is_pmjdy"]    = (df["scheme_code"] == "PMJDY").astype(int)
    df["is_regular"]  = (df["scheme_code"] == "REGULAR").astype(int)
    df["is_govt_scheme"] = (df["scheme_code"].isin(
        ["PMJDY", "PMSBY", "PMJJBY", "APY", "SCSS", "SSA"])).astype(int)
    return df[["account_id", "is_pmjdy", "is_regular", "is_govt_scheme"]]


def build_branch_features(df_branch: pd.DataFrame) -> pd.DataFrame:
    df = df_branch.copy()
    # README columns: branch_code, branch_address, branch_pin_code, branch_city,
    # branch_state, branch_employee_count, branch_turnover, branch_asset_size, branch_type
    df["branch_type_enc"] = df["branch_type"].map(
        {"urban": 2, "semi-urban": 1, "rural": 0}).fillna(1).astype(int)
    df["branch_turnover_log"]  = np.log1p(df["branch_turnover"].clip(lower=0).fillna(0))
    df["branch_asset_log"]     = np.log1p(df["branch_asset_size"].clip(lower=0).fillna(0))
    df["branch_employee_count"] = df["branch_employee_count"].fillna(0)
    return df[["branch_code", "branch_type_enc", "branch_employee_count",
               "branch_turnover_log", "branch_asset_log"]]


def run() -> pd.DataFrame:
    log.info("=== Phase 1: Static Feature Engineering ===")

    df_train  = read_static(CFG.paths.train_labels)
    df_acc    = read_static(CFG.paths.accounts)
    df_cust   = read_static(CFG.paths.customers)
    df_link   = read_static(CFG.paths.customer_account_linkage)
    df_prod   = read_static(CFG.paths.product_details)
    df_demo   = read_static(CFG.paths.demographics)
    df_scheme = read_static(CFG.paths.accounts_add)
    df_branch = read_static(CFG.paths.branch)

    acc_feat    = build_account_features(df_acc, df_train)
    cust_feat   = build_customer_features(df_cust)
    prod_feat   = build_product_features(df_prod)
    demo_feat   = build_demographics_features(df_demo)
    scheme_feat = build_scheme_features(df_scheme)
    branch_feat = build_branch_features(df_branch)

    # Merge: account → customer via linkage table
    df = acc_feat.merge(df_link[["account_id", "customer_id"]], on="account_id", how="left")
    df = df.merge(cust_feat,   on="customer_id", how="left")
    df = df.merge(prod_feat,   on="customer_id", how="left")
    df = df.merge(demo_feat,   on="customer_id", how="left")
    df = df.merge(scheme_feat, on="account_id",  how="left")
    df = df.merge(branch_feat, on="branch_code", how="left")

    df.drop(columns=["customer_id", "branch_code"], inplace=True, errors="ignore")

    # Fill remaining NaNs
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    log.info(f"Static features shape: {df.shape}")
    save_features(df, CFG.paths.static_features)
    return df


if __name__ == "__main__":
    run()
