"""
Shared pytest fixtures for the Telco Churn test suite.

These fixtures are available to ALL test files without explicit import.
Slide 25 (Lab 2, Scenario C): changing tenure dtype here to str triggers
pandera.errors.SchemaError — exactly as demonstrated in the lab.
"""

from __future__ import annotations

import pandas as pd
import pytest


# ── Sample DataFrames ─────────────────────────────────────────────────────────

@pytest.fixture()
def sample_raw_df() -> pd.DataFrame:
    """5-row Telco Churn DataFrame with correct dtypes.

    Used in unit tests for encode_features() and Pandera schema validation.

    LAB 2 Scenario C: change  tenure: int  →  tenure: str  here to trigger
    pandera.errors.SchemaError: Expected int64, got object.
    """
    return pd.DataFrame({
        "customerID":      ["0001", "0002", "0003", "0004", "0005"],
        "tenure":          [1,  24, 60, 12,  6],         # int  ← DO NOT change
        "MonthlyCharges":  [29.85, 56.95, 42.30, 89.10, 20.05],
        "TotalCharges":    [29.85, 1889.50, 2320.80, 1138.80, 172.70],
        "SeniorCitizen":   [0, 0, 1, 0, 0],
        "Contract":        ["Month-to-month", "One year", "Two year",
                            "Month-to-month", "Month-to-month"],
        "PaymentMethod":   ["Electronic check", "Bank transfer (automatic)",
                            "Credit card (automatic)", "Mailed check",
                            "Electronic check"],
        "InternetService": ["Fiber optic", "DSL", "Fiber optic", "DSL", "No"],
        "OnlineSecurity":  ["No", "Yes", "Yes", "No", "No"],
        "TechSupport":     ["No", "Yes", "No", "Yes", "No"],
        "Churn":           ["Yes", "No", "No", "Yes", "Yes"],
    })


@pytest.fixture()
def high_risk_customer() -> dict:
    """Low tenure, month-to-month → expected high churn probability (> 0.5).

    Used in integration tests (slide 17, L2 Model Behaviour).
    """
    return {
        "tenure": 1,
        "MonthlyCharges": 85.0,
        "TotalCharges": 85.0,
        "SeniorCitizen": 0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "Churn": "No",
    }


@pytest.fixture()
def low_risk_customer() -> dict:
    """Long tenure, two-year contract → expected low churn probability (< 0.4).

    Used in integration tests (slide 17, L2 Model Behaviour).
    """
    return {
        "tenure": 60,
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0,
        "SeniorCitizen": 0,
        "Contract": "Two year",
        "PaymentMethod": "Bank transfer (automatic)",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
        "Churn": "No",
    }
