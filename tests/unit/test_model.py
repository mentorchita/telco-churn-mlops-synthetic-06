"""
Model quality gate tests  (Prompt 2 from slide 26).

These tests train the XGBoost model ONCE (scope='module') and validate
quality thresholds — exactly as described on slide 12.

LAB 2 Scenario A: change  MIN_AUC = 0.99  to trigger
    AssertionError: 0.887 < 0.99
Then restore  MIN_AUC = 0.82  to fix.
"""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features.build_features import get_X_y

# ── Quality gate thresholds (slide 12) ───────────────────────────────────────
# LAB 2 Scenario A: change this to 0.99 → AssertionError, then restore 0.82
MIN_AUC       = 0.82
MIN_F1        = 0.70
MIN_PRECISION = 0.75
EXPECTED_FEATURE_COUNT = 19   # slide 12: "Feature count = 19"

DATA_PATH = "data/raw/telco_train.csv"


# ── Module-scoped fixture: train model ONCE ───────────────────────────────────

@pytest.fixture(scope="module")
def trained_model_artifacts():
    """Train model once for the entire test session.

    scope='module' = train once, reuse in all tests below.
    Without this, each test re-trains → 6 × 2 min = 12 min.
    With this, total time = ~2 min.
    """
    df = pd.read_csv(DATA_PATH)
    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, verbose=False)

    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)

    return {
        "model":    model,
        "dummy":    dummy,
        "X_train":  X_train,
        "X_test":   X_test,
        "y_train":  y_train,
        "y_test":   y_test,
        "n_features": X.shape[1],
    }


# ── Quality gate tests ────────────────────────────────────────────────────────

@pytest.mark.slow
class TestModelQualityGate:
    """All tests on slide 12 mapped to pytest."""

    def test_auc_above_threshold(self, trained_model_artifacts):
        art = trained_model_artifacts
        y_proba = art["model"].predict_proba(art["X_test"])[:, 1]
        auc = roc_auc_score(art["y_test"], y_proba)
        mlflow.log_metric("test_roc_auc", auc)
        assert auc >= MIN_AUC, (
            f"ROC-AUC {auc:.3f} < threshold {MIN_AUC}. "
            f"Model quality degraded — check recent feature changes."
        )

    def test_f1_above_threshold(self, trained_model_artifacts):
        art = trained_model_artifacts
        y_pred = art["model"].predict(art["X_test"])
        f1 = f1_score(art["y_test"], y_pred)
        mlflow.log_metric("test_f1_score", f1)
        assert f1 >= MIN_F1, f"F1 {f1:.3f} < threshold {MIN_F1}"

    def test_precision_above_threshold(self, trained_model_artifacts):
        art = trained_model_artifacts
        y_pred = art["model"].predict(art["X_test"])
        prec = precision_score(art["y_test"], y_pred)
        mlflow.log_metric("test_precision", prec)
        assert prec >= MIN_PRECISION, f"Precision {prec:.3f} < threshold {MIN_PRECISION}"

    def test_beats_dummy_classifier_auc(self, trained_model_artifacts):
        """Model must outperform DummyClassifier — sanity check (slide 12)."""
        art = trained_model_artifacts
        model_auc = roc_auc_score(
            art["y_test"],
            art["model"].predict_proba(art["X_test"])[:, 1]
        )
        dummy_auc = roc_auc_score(
            art["y_test"],
            art["dummy"].predict_proba(art["X_test"])[:, 1]
        )
        assert model_auc > dummy_auc, (
            f"Model AUC {model_auc:.3f} does not beat DummyClassifier {dummy_auc:.3f}"
        )

    def test_beats_dummy_classifier_f1(self, trained_model_artifacts):
        art = trained_model_artifacts
        model_f1 = f1_score(art["y_test"], art["model"].predict(art["X_test"]))
        dummy_f1 = f1_score(art["y_test"], art["dummy"].predict(art["X_test"]),
                            zero_division=0)
        assert model_f1 > dummy_f1, (
            f"Model F1 {model_f1:.3f} does not beat DummyClassifier {dummy_f1:.3f}"
        )

    def test_predict_proba_in_range(self, trained_model_artifacts):
        """All probabilities must be in [0, 1] (slide 17 L2 check)."""
        art = trained_model_artifacts
        proba = art["model"].predict_proba(art["X_test"])[:, 1]
        assert (proba >= 0.0).all(), "Probability < 0 found"
        assert (proba <= 1.0).all(), "Probability > 1 found"

    def test_feature_count(self, trained_model_artifacts):
        """Feature count must equal 19 — catches silent feature additions (slide 12)."""
        n = trained_model_artifacts["n_features"]
        assert n == EXPECTED_FEATURE_COUNT, (
            f"Feature count {n} != expected {EXPECTED_FEATURE_COUNT}. "
            f"Someone added or removed a feature — check build_features.py"
        )
