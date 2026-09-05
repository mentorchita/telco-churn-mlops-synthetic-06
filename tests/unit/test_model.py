"""
Model quality gate tests  (Prompt 2 from slide 26).

These tests train the XGBoost model ONCE (scope='module') and validate
quality thresholds — exactly as described on slide 12.

LAB 2 Scenario A: change  MIN_AUC = 0.99  to trigger
    AssertionError: 0.887 < 0.99
Then restore  MIN_AUC = 0.82  to fix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMN, get_X_y

# ── Quality gate thresholds (slide 12) ───────────────────────────────────────
# LAB 2 Scenario A: change this to 0.99 → AssertionError, then restore 0.82
MIN_AUC = 0.82
MIN_F1 = 0.70
MIN_PRECISION = 0.75
# X excludes the target column from FEATURE_COLUMNS
EXPECTED_FEATURE_COUNT = len([c for c in FEATURE_COLUMNS if c != TARGET_COLUMN])

DATA_PATH = "data/raw/telco_train.csv"


def _make_synthetic_train_df(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Telco data when CSV is absent (CI / local without data).

    Risk factors mirror the real generator so XGBoost clears quality gates.
    """
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 73, size=n)
    monthly = rng.uniform(18.0, 120.0, size=n)
    total = monthly * np.maximum(tenure, 1) * rng.uniform(0.95, 1.05, size=n)
    senior = rng.integers(0, 2, size=n)

    contracts = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.55, 0.25, 0.20],
    )
    payments = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
        p=[0.40, 0.15, 0.25, 0.20],
    )
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.30, 0.50, 0.20])
    online_sec = np.where(internet == "No", "No", rng.choice(["Yes", "No"], size=n))
    tech = np.where(internet == "No", "No", rng.choice(["Yes", "No"], size=n))

    logit = (
        -2.2
        + 2.8 * (contracts == "Month-to-month")
        + 1.6 * (payments == "Electronic check")
        + 1.0 * (internet == "Fiber optic")
        + 2.2 * (tenure < 6)
        + 1.2 * (tenure < 12)
        - 1.8 * (contracts == "Two year")
        - 1.0 * (online_sec == "Yes")
        - 1.0 * (tech == "Yes")
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = np.where(rng.random(n) < prob, "Yes", "No")

    return pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(n)],
            "tenure": tenure.astype(int),
            "MonthlyCharges": np.round(monthly, 2),
            "TotalCharges": np.round(total, 2),
            "SeniorCitizen": senior.astype(int),
            "Contract": contracts,
            "PaymentMethod": payments,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "TechSupport": tech,
            "Churn": churn,
        }
    )


def _load_train_df() -> pd.DataFrame:
    """Load training CSV if present; otherwise synthesize data for CI."""
    path = Path(DATA_PATH)
    if path.exists():
        return pd.read_csv(path)
    return _make_synthetic_train_df()


@pytest.fixture(scope="module")
def trained_model_artifacts():
    """Train model once for the entire test session.

    scope='module' = train once, reuse in all tests below.
    """
    df = _load_train_df()
    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, verbose=False)

    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)

    return {
        "model": model,
        "dummy": dummy,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "n_features": X.shape[1],
    }


@pytest.mark.slow
class TestModelQualityGate:
    """Quality gate thresholds from slide 12."""

    def test_auc_above_threshold(self, trained_model_artifacts):
        art = trained_model_artifacts
        proba = art["model"].predict_proba(art["X_test"])[:, 1]
        auc = roc_auc_score(art["y_test"], proba)
        assert auc >= MIN_AUC, f"AUC {auc:.3f} < threshold {MIN_AUC}"

    def test_f1_above_threshold(self, trained_model_artifacts):
        art = trained_model_artifacts
        pred = art["model"].predict(art["X_test"])
        f1 = f1_score(art["y_test"], pred)
        assert f1 >= MIN_F1, f"F1 {f1:.3f} < threshold {MIN_F1}"

    def test_precision_above_threshold(self, trained_model_artifacts):
        art = trained_model_artifacts
        pred = art["model"].predict(art["X_test"])
        prec = precision_score(art["y_test"], pred)
        assert prec >= MIN_PRECISION, f"Precision {prec:.3f} < threshold {MIN_PRECISION}"

    def test_beats_dummy_classifier_auc(self, trained_model_artifacts):
        art = trained_model_artifacts
        model_auc = roc_auc_score(art["y_test"], art["model"].predict_proba(art["X_test"])[:, 1])
        dummy_auc = roc_auc_score(art["y_test"], art["dummy"].predict_proba(art["X_test"])[:, 1])
        assert (
            model_auc > dummy_auc
        ), f"Model AUC {model_auc:.3f} does not beat DummyClassifier {dummy_auc:.3f}"

    def test_beats_dummy_classifier_f1(self, trained_model_artifacts):
        art = trained_model_artifacts
        model_f1 = f1_score(art["y_test"], art["model"].predict(art["X_test"]))
        dummy_f1 = f1_score(art["y_test"], art["dummy"].predict(art["X_test"]), zero_division=0)
        assert (
            model_f1 > dummy_f1
        ), f"Model F1 {model_f1:.3f} does not beat DummyClassifier {dummy_f1:.3f}"

    def test_predict_proba_in_range(self, trained_model_artifacts):
        """All probabilities must be in [0, 1] (slide 17 L2 check)."""
        art = trained_model_artifacts
        proba = art["model"].predict_proba(art["X_test"])[:, 1]
        assert (proba >= 0.0).all(), "Probability < 0 found"
        assert (proba <= 1.0).all(), "Probability > 1 found"

    def test_feature_count(self, trained_model_artifacts):
        """Feature count must match FEATURE_COLUMNS without target."""
        n = trained_model_artifacts["n_features"]
        assert n == EXPECTED_FEATURE_COUNT, (
            f"Feature count {n} != expected {EXPECTED_FEATURE_COUNT}. "
            f"Someone added or removed a feature — check build_features.py"
        )
