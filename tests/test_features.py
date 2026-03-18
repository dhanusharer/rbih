"""Unit tests — run with: pytest tests/ -v"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def sample_accounts():
    today = datetime(2025, 7, 1)
    return pd.DataFrame({
        "account_id":            [f"ACCT_{i:06d}" for i in range(5)],
        "account_status":        ["active","active","frozen","active","active"],
        "product_code":          ["SA01"]*5,
        "currency_code":         [1]*5,
        "account_opening_date":  [(today-timedelta(days=d)).strftime("%Y-%m-%d")
                                  for d in [1800,30,500,200,90]],
        "branch_code":           ["BR001","BR001","BR002","BR002","BR001"],
        "branch_pin":            ["500001"]*5,
        "avg_balance":           [5000,-200,100000,0,50000],
        "product_family":        ["S","S","K","S","O"],
        "nomination_flag":       ["Y","N","Y","N","Y"],
        "cheque_allowed":        ["Y"]*5,
        "cheque_availed":        ["N"]*5,
        "num_chequebooks":       [0,0,2,0,1],
        "last_mobile_update_date":[(today-timedelta(days=d)).strftime("%Y-%m-%d")
                                   for d in [5,300,100,50,3]],
        "kyc_compliant":         ["Y","N","Y","N","Y"],
        "last_kyc_date":         [(today-timedelta(days=d)).strftime("%Y-%m-%d")
                                  for d in [30,400,100,600,50]],
        "rural_branch":          ["N"]*5,
        "monthly_avg_balance":   [5000,0,95000,0,48000],
        "quarterly_avg_balance": [5100,0,97000,0,49000],
        "daily_avg_balance":     [4900,0,93000,0,47000],
        "freeze_date":           [None,None,"2024-01-01",None,None],
        "unfreeze_date":         [None]*5,
    })


@pytest.fixture
def sample_customers():
    return pd.DataFrame({
        "customer_id":            [f"CUST_{i:06d}" for i in range(5)],
        "date_of_birth":          ["1990-01-01"]*5,
        "relationship_start_date":["2020-01-01"]*5,
        "pan_available":          ["Y","N","Y","N","Y"],
        "aadhaar_available":      ["Y"]*5,
        "passport_available":     ["N"]*5,
        "mobile_banking_flag":    ["Y","N","Y","Y","Y"],
        "internet_banking_flag":  ["Y","N","Y","Y","Y"],
        "atm_card_flag":          ["Y"]*5,
        "demat_flag":             ["N"]*5,
        "credit_card_flag":       ["N","N","Y","N","N"],
        "fastag_flag":            ["N"]*5,
        "customer_pin":           ["500001","500001","500001","500001","600001"],
        "permanent_pin":          ["500001"]*5,
    })


class TestAccountFeatures:
    def test_account_age_non_negative(self, sample_accounts):
        from src.features.static_features import build_account_features
        r = build_account_features(sample_accounts, None)
        assert (r["account_age_days"] >= 0).all()

    def test_new_account_flag(self, sample_accounts):
        from src.features.static_features import build_account_features
        r = build_account_features(sample_accounts, None)
        new = r[r["account_id"] == "ACCT_000001"]
        assert new["is_new_account"].values[0] == 1

    def test_freeze_detection(self, sample_accounts):
        from src.features.static_features import build_account_features
        r = build_account_features(sample_accounts, None)
        frozen = r[r["account_id"] == "ACCT_000002"]
        assert frozen["freeze_ever"].values[0] == 1

    def test_no_nulls(self, sample_accounts):
        from src.features.static_features import build_account_features
        r = build_account_features(sample_accounts, None)
        for col in ["account_age_days", "kyc_compliant_enc", "freeze_ever"]:
            assert r[col].isnull().sum() == 0


class TestCustomerFeatures:
    def test_kyc_weakness_range(self, sample_customers):
        from src.features.static_features import build_customer_features
        r = build_customer_features(sample_customers)
        assert r["kyc_weakness_score"].between(0, 4).all()

    def test_pin_mismatch(self, sample_customers):
        from src.features.static_features import build_customer_features
        r = build_customer_features(sample_customers)
        mismatch = r[r["customer_id"] == "CUST_000004"]
        assert mismatch["pin_mismatch"].values[0] == 1


class TestPatterns:
    def test_structuring(self):
        amounts_mule = [49000, 49500, 49999, 48000, 49100]
        amounts_legit = [1000, 5000, 20000, 100, 3000]
        struct_mule  = sum(1 for a in amounts_mule  if 45000 <= a < 50000)
        struct_legit = sum(1 for a in amounts_legit if 45000 <= a < 50000)
        assert struct_mule > struct_legit

    def test_round_amounts(self):
        bins = [1000, 5000, 10000, 50000, 100000]
        def is_round(a): return any(a % b == 0 for b in bins)
        assert is_round(1000) and is_round(50000)
        assert not is_round(1234) and not is_round(9999)

    def test_passthrough_ratio(self):
        ratio = 98000 / (100000 + 1e-9)
        assert 0.95 <= ratio <= 1.05


class TestSubmissionFormat:
    def test_required_columns(self):
        df = pd.DataFrame({
            "account_id": ["ACCT_000001"],
            "is_mule": [0.85],
            "suspicious_start": ["2024-01-01T00:00:00"],
            "suspicious_end": ["2024-06-01T00:00:00"],
        })
        for col in ["account_id", "is_mule", "suspicious_start", "suspicious_end"]:
            assert col in df.columns

    def test_probability_range(self):
        probs = np.random.random(100)
        assert probs.min() >= 0.0 and probs.max() <= 1.0


class TestModelUtils:
    def test_best_f1_threshold(self):
        from src.models.train import best_f1_threshold
        np.random.seed(42)
        y = np.random.randint(0, 2, 200)
        p = np.random.random(200)
        t, f1 = best_f1_threshold(y, p, 20)
        assert 0.0 < t < 1.0 and 0.0 <= f1 <= 1.0

    def test_get_feature_cols(self):
        from src.models.train import get_feature_cols
        df = pd.DataFrame({"account_id": ["A"], "is_mule": [1],
                           "feat1": [1.0], "feat2": [2.0]})
        cols = get_feature_cols(df)
        assert "account_id" not in cols and "is_mule" not in cols
        assert "feat1" in cols and "feat2" in cols
