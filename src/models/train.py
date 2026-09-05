"""
Model training script for Telco Churn prediction.

Trains an XGBoost classifier, validates quality gates, and registers
the model in MLflow Model Registry.

Usage:
    python -m src.models.train                      # default config
    python -m src.models.train --data data/raw/telco_train.csv
    python -m src.models.train --run-name my-run

Slide 11 quality gate:
    - ROC-AUC  >= 0.82
    - F1-Score >= 0.70
    - Precision >= 0.75
    - Feature count == 19
    - Beats DummyClassifier on all metrics
"""

from __future__ import annotations

import argparse
import logging
import sys
import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features.build_features import get_X_y

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Quality gate thresholds (slide 12) ───────────────────────────────────────
MIN_AUC = 0.82
MIN_F1 = 0.70
MIN_PRECISION = 0.75
EXPECTED_FEATURES = 19   # must equal len(FEATURE_COLUMNS) - 1 (exclude target)

# ── Hyperparameters ────────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
    "use_label_encoder": False,
    "eval_metric":      "logloss",
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def quality_gate(auc: float, f1: float, prec: float,
                 dummy_auc: float, dummy_f1: float, dummy_prec: float,
                 n_features: int) -> None:
    """Assert all quality thresholds.  Exit with code 1 on failure (slide 12)."""
    failures = []

    if auc < MIN_AUC:
        failures.append(f"AUC {auc:.3f} < threshold {MIN_AUC}")
    if f1 < MIN_F1:
        failures.append(f"F1 {f1:.3f} < threshold {MIN_F1}")
    if prec < MIN_PRECISION:
        failures.append(f"Precision {prec:.3f} < threshold {MIN_PRECISION}")
    if auc <= dummy_auc:
        failures.append(f"AUC {auc:.3f} does not beat DummyClassifier {dummy_auc:.3f}")
    if f1 <= dummy_f1:
        failures.append(f"F1 {f1:.3f} does not beat DummyClassifier {dummy_f1:.3f}")
    if n_features != EXPECTED_FEATURES:
        failures.append(f"Feature count {n_features} != expected {EXPECTED_FEATURES}")

    if failures:
        logger.error("QUALITY GATE FAILED:")
        for msg in failures:
            logger.error("  ✗ %s", msg)
        sys.exit(1)

    logger.info("✓ All quality gates passed")


def train(data_path: str = "data/raw/telco_train.csv",
          run_name: str = "telco-churn-train",
          register: bool = True) -> None:

    df = load_data(data_path)
    X, y = get_X_y(df)

    n_features = X.shape[1]
    logger.info("Feature matrix shape: %s", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(XGBOOST_PARAMS)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test",  len(X_test))

        # ── Train XGBoost ─────────────────────────────────────────────────
        model = XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        # ── Evaluate ──────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)

        mlflow.log_metric("roc_auc",   auc)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("precision", prec)

        logger.info("ROC-AUC=%.3f  F1=%.3f  Precision=%.3f", auc, f1, prec)

        # ── Dummy baseline ────────────────────────────────────────────────
        dummy = DummyClassifier(strategy="most_frequent", random_state=42)
        dummy.fit(X_train, y_train)
        d_pred = dummy.predict(X_test)
        d_proba = dummy.predict_proba(X_test)[:, 1]
        d_auc = roc_auc_score(y_test, d_proba)
        d_f1 = f1_score(y_test, d_pred, zero_division=0)
        d_prec = precision_score(y_test, d_pred, zero_division=0)

        mlflow.log_metric("dummy_roc_auc",   d_auc)
        mlflow.log_metric("dummy_f1_score",  d_f1)
        mlflow.log_metric("dummy_precision", d_prec)

        # ── Quality gate (exits with 1 on failure) ────────────────────────
        quality_gate(auc, f1, prec, d_auc, d_f1, d_prec, n_features)

        # ── Log model ─────────────────────────────────────────────────────
        signature = mlflow.models.infer_signature(
            X_train, model.predict_proba(X_train)[:, 1]
        )
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="telco-churn-prod" if register else None,
            signature=signature,
            input_example=X_train[:3],
        )
        logger.info("Model logged to MLflow run %s", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Telco Churn model")
    parser.add_argument("--data",     default="data/raw/telco_train.csv")
    parser.add_argument("--run-name", default="telco-churn-train")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    train(data_path=args.data,
          run_name=args.run_name,
          register=not args.no_register)
