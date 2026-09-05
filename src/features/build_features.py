"""
Feature engineering for the Telco Churn MLOps project.

This module contains all data transformation logic used both during
training and at serving time.  Any change here affects both pipelines.

Slide 11 (lint-and-test): this file is tested by tests/unit/test_features.py
Slide 17 (integration tests - L4 Feature Parity): serving must use exactly
    the same encoding as training — both import from here.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── Canonical feature list ────────────────────────────────────────────────────
# IMPORTANT: changing this list breaks serving (slide 12 quality gate catches
# this via the "Feature count = 19" test).
FEATURE_COLUMNS: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "Contract_Month-to-month",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "InternetService_DSL",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No",
    "OnlineSecurity_Yes",
    "TechSupport_No",
    "TechSupport_Yes",
    "Churn_encoded",
]

TARGET_COLUMN = "Churn_encoded"

# Contract encoding map (used in serving to avoid fit_transform inconsistency)
CONTRACT_DUMMIES = ["Month-to-month", "One year", "Two year"]
PAYMENT_DUMMIES = [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]
INTERNET_DUMMIES = ["DSL", "Fiber optic", "No"]
ONLINE_SECURITY_DUMMIES = ["No", "Yes"]
TECH_SUPPORT_DUMMIES = ["No", "Yes"]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode raw Telco Churn DataFrame into model-ready features.

    Applies one-hot encoding to categorical columns and label encoding
    to the target variable.  Does NOT modify the original DataFrame.

    Args:
        df: Raw DataFrame with Telco Churn columns.

    Returns:
        New DataFrame with exactly the columns in FEATURE_COLUMNS.

    Raises:
        ValueError: If required columns are missing from ``df``.
    """
    required = {
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
        "Contract",
        "PaymentMethod",
        "InternetService",
        "OnlineSecurity",
        "TechSupport",
        "Churn",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Work on a copy — never mutate the caller's DataFrame
    out = df.copy()

    # Fix TotalCharges: sometimes arrives as string with spaces
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["TotalCharges"] = out["TotalCharges"].fillna(out["TotalCharges"].median())

    # One-hot encode categoricals (drop_first=False for explicit column names)
    for col, categories in [
        ("Contract", CONTRACT_DUMMIES),
        ("PaymentMethod", PAYMENT_DUMMIES),
        ("InternetService", INTERNET_DUMMIES),
        ("OnlineSecurity", ONLINE_SECURITY_DUMMIES),
        ("TechSupport", TECH_SUPPORT_DUMMIES),
    ]:
        dummies = pd.get_dummies(out[col], prefix=col)
        # Ensure all expected columns present even if category absent in df
        for cat in categories:
            col_name = f"{col}_{cat}"
            if col_name not in dummies.columns:
                dummies[col_name] = 0
        out = pd.concat([out, dummies], axis=1)

    # Label-encode target: Yes -> 1, No -> 0
    le = LabelEncoder()
    out["Churn_encoded"] = le.fit_transform(out["Churn"].astype(str))

    # Select and order exactly FEATURE_COLUMNS (minus target for X, included for full df)
    feature_cols_present = [c for c in FEATURE_COLUMNS if c in out.columns]
    out = out[feature_cols_present]

    logger.debug("encode_features: output shape %s", out.shape)
    return out


def get_X_y(df: pd.DataFrame):
    """Return (X, y) numpy arrays from an encoded DataFrame."""
    encoded = encode_features(df)
    X_cols = [c for c in FEATURE_COLUMNS if c != TARGET_COLUMN]
    X = encoded[X_cols].values
    y = encoded[TARGET_COLUMN].values
    return X, y
