"""Train a small fallback model for API image (CI / local without MLflow)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features.build_features import get_X_y

OUT = Path("models/churn_model.joblib")


def make_df(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, size=n)
    monthly = rng.uniform(18.0, 120.0, size=n)
    total = monthly * np.maximum(tenure, 1) * rng.uniform(0.95, 1.05, size=n)
    senior = rng.integers(0, 2, size=n)
    contracts = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]
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


def main() -> None:
    df = make_df()
    X, y = get_X_y(df)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, verbose=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT)
    print(f"Saved fallback model → {OUT}")


if __name__ == "__main__":
    main()
