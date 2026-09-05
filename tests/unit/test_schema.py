"""
Pandera data schema validation tests  (Prompt 4 from slide 27).

This module must run as the FIRST CI step — before any model training.
It catches broken input data before it reaches the training loop.

LAB 2 Scenario C: change  tenure: int  →  tenure: str  in conftest.py
    to trigger:  pandera.errors.SchemaError: Expected int64, got object
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
import pytest
from pandera import Column, DataFrameSchema, Check

# ── Pandera schema definition ─────────────────────────────────────────────────

TELCO_SCHEMA = DataFrameSchema(
    {
        "tenure": Column(
            int,
            checks=[Check.ge(0), Check.le(72)],
            nullable=False,
            description="Months as customer",
        ),
        "MonthlyCharges": Column(
            float,
            checks=[Check.ge(0.0), Check.le(300.0)],
            nullable=False,
        ),
        "TotalCharges": Column(
            float,
            checks=[Check.ge(0.0)],
            nullable=True,  # may be missing for new customers
        ),
        "SeniorCitizen": Column(
            int,
            checks=[Check.isin([0, 1])],
            nullable=False,
        ),
        "Contract": Column(
            str,
            checks=[Check.isin(["Month-to-month", "One year", "Two year"])],
            nullable=False,
        ),
        "PaymentMethod": Column(
            str,
            checks=[
                Check.isin(
                    [
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                        "Electronic check",
                        "Mailed check",
                    ]
                )
            ],
            nullable=False,
        ),
        "InternetService": Column(
            str,
            checks=[Check.isin(["DSL", "Fiber optic", "No"])],
            nullable=False,
        ),
        "OnlineSecurity": Column(
            str,
            checks=[Check.isin(["No", "Yes", "No internet service"])],
            nullable=False,
        ),
        "TechSupport": Column(
            str,
            checks=[Check.isin(["No", "Yes", "No internet service"])],
            nullable=False,
        ),
        "Churn": Column(
            str,
            checks=[Check.isin(["Yes", "No"])],
            nullable=False,
        ),
    },
    coerce=False,  # strict — do not auto-cast types
)


def validate_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw Telco Churn DataFrame against the schema.

    Raises:
        pandera.errors.SchemaError: if any constraint is violated.
    """
    return TELCO_SCHEMA.validate(df, lazy=True)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestValidTelcoData:
    """Valid data must pass without exceptions."""

    def test_valid_sample_passes(self, sample_raw_df):
        validated = validate_telco_data(sample_raw_df)
        assert validated is not None
        assert len(validated) == len(sample_raw_df)


class TestInvalidData:
    """Invalid data must raise SchemaError with clear messages."""

    def test_negative_tenure_raises(self, sample_raw_df):
        bad = sample_raw_df.copy()
        bad.loc[0, "tenure"] = -1
        with pytest.raises(pa.errors.SchemaError):
            validate_telco_data(bad)

    def test_tenure_over_72_raises(self, sample_raw_df):
        bad = sample_raw_df.copy()
        bad.loc[0, "tenure"] = 100
        with pytest.raises(pa.errors.SchemaError):
            validate_telco_data(bad)

    def test_null_monthly_charges_raises(self, sample_raw_df):
        bad = sample_raw_df.copy()
        bad.loc[0, "MonthlyCharges"] = None
        with pytest.raises(pa.errors.SchemaError):
            validate_telco_data(bad)

    def test_bad_contract_value_raises(self, sample_raw_df):
        bad = sample_raw_df.copy()
        bad.loc[0, "Contract"] = "Quarterly"  # not in allowed set
        with pytest.raises(pa.errors.SchemaError):
            validate_telco_data(bad)

    def test_bad_churn_value_raises(self, sample_raw_df):
        bad = sample_raw_df.copy()
        bad.loc[0, "Churn"] = "Maybe"
        with pytest.raises(pa.errors.SchemaError):
            validate_telco_data(bad)

    def test_wrong_tenure_type_raises(self, sample_raw_df):
        """LAB 2 Scenario C: this is what changing tenure to str triggers."""
        bad = sample_raw_df.copy()
        bad["tenure"] = bad["tenure"].astype(str)  # str instead of int
        with pytest.raises(pa.errors.SchemaError):
            validate_telco_data(bad)


class TestMissingColumn:
    """Missing columns produce clear error messages (slide 27, Prompt 4)."""

    def test_missing_tenure_raises(self, sample_raw_df):
        df = sample_raw_df.drop(columns=["tenure"])
        with pytest.raises(pa.errors.SchemaError) as exc_info:
            validate_telco_data(df)
        # Error message should mention the missing column
        assert "tenure" in str(exc_info.value).lower()
