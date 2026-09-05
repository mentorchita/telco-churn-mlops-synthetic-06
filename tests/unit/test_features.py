"""
Unit tests for src/features/build_features.py  (Prompt 1 from slide 26).

Tests: happy-path, edge cases, immutability, type assertions, parametrize.
These run WITHOUT model training — fast (<5 s total).
"""

from __future__ import annotations

import numpy as np  # used in TestGetXY.test_returns_numpy_arrays
import pandas as pd
import pytest

from src.features.build_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    encode_features,
    get_X_y,
)

# Expected output columns (excluding target for X)
X_COLUMNS = [c for c in FEATURE_COLUMNS if c != TARGET_COLUMN]
N_FEATURES = len(X_COLUMNS)   # must be 19


# ── Happy-path ─────────────────────────────────────────────────────────────────

class TestEncodeFeatures:
    """Core behaviour of encode_features()."""

    def test_returns_dataframe(self, sample_raw_df):
        result = encode_features(sample_raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns_match_feature_list(self, sample_raw_df):
        result = encode_features(sample_raw_df)
        assert set(result.columns) == set(FEATURE_COLUMNS)

    def test_feature_count_is_19(self, sample_raw_df):
        """Feature count gate — matches slide 12 quality-gate assertion."""
        result = encode_features(sample_raw_df)
        x_cols = [c for c in result.columns if c != TARGET_COLUMN]
        assert len(x_cols) == N_FEATURES, (
            f"Expected {N_FEATURES} features, got {len(x_cols)}: {x_cols}"
        )

    def test_churn_encoded_is_binary(self, sample_raw_df):
        result = encode_features(sample_raw_df)
        assert result["Churn_encoded"].isin([0, 1]).all()

    def test_churn_yes_encoded_as_1(self, sample_raw_df):
        result = encode_features(sample_raw_df)
        # row 0 has Churn='Yes'
        assert result["Churn_encoded"].iloc[0] == 1

    def test_churn_no_encoded_as_0(self, sample_raw_df):
        result = encode_features(sample_raw_df)
        # row 1 has Churn='No'
        assert result["Churn_encoded"].iloc[1] == 0


# ── Immutability ───────────────────────────────────────────────────────────────

class TestImmutability:
    """encode_features must NOT modify the caller's DataFrame (slide 26, req 6)."""

    def test_original_df_not_modified(self, sample_raw_df):
        original_cols = list(sample_raw_df.columns)
        original_tenure = sample_raw_df["tenure"].tolist()
        encode_features(sample_raw_df)
        assert list(sample_raw_df.columns) == original_cols
        assert sample_raw_df["tenure"].tolist() == original_tenure

    def test_original_df_shape_unchanged(self, sample_raw_df):
        original_shape = sample_raw_df.shape
        encode_features(sample_raw_df)
        assert sample_raw_df.shape == original_shape


# ── Contract one-hot encoding ─────────────────────────────────────────────────

@pytest.mark.parametrize("contract,expected_col", [
    ("Month-to-month", "Contract_Month-to-month"),
    ("One year",       "Contract_One year"),
    ("Two year",       "Contract_Two year"),
])
def test_contract_one_hot(sample_raw_df, contract, expected_col):
    """Each contract type maps to exactly one column = 1 (slide 26, req 4)."""
    row = sample_raw_df.copy()
    row["Contract"] = contract
    result = encode_features(row.iloc[:1])
    assert expected_col in result.columns
    assert result[expected_col].iloc[0] == 1
    # Other contract columns must be 0
    other_contract_cols = [
        c for c in result.columns
        if c.startswith("Contract_") and c != expected_col
    ]
    for c in other_contract_cols:
        assert result[c].iloc[0] == 0, f"{c} should be 0 for contract={contract}"


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Slide 26 requirements 3: edge cases."""

    def test_all_contracts_identical(self, sample_raw_df):
        df = sample_raw_df.copy()
        df["Contract"] = "Month-to-month"
        result = encode_features(df)
        assert result["Contract_Month-to-month"].all()
        assert (result["Contract_One year"] == 0).all()
        assert (result["Contract_Two year"] == 0).all()

    def test_nan_in_total_charges_imputed(self, sample_raw_df):
        df = sample_raw_df.copy()
        df.loc[0, "TotalCharges"] = None
        # Should not raise — NaN is filled with median
        result = encode_features(df)
        assert not result.isnull().any().any()

    def test_missing_required_column_raises(self, sample_raw_df):
        df = sample_raw_df.drop(columns=["tenure"])
        with pytest.raises(ValueError, match="Missing required columns"):
            encode_features(df)


# ── get_X_y ───────────────────────────────────────────────────────────────────

class TestGetXY:
    """get_X_y returns numpy arrays of correct shape."""

    def test_X_shape(self, sample_raw_df):
        X, y = get_X_y(sample_raw_df)
        assert X.shape == (len(sample_raw_df), N_FEATURES)

    def test_y_is_binary(self, sample_raw_df):
        X, y = get_X_y(sample_raw_df)
        assert set(y).issubset({0, 1})

    def test_returns_numpy_arrays(self, sample_raw_df):
        X, y = get_X_y(sample_raw_df)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
