"""
Smoke tests — run immediately after every staging deploy (slide 15).

These are intentionally minimal and fast (<30 s total).
A failing smoke test triggers an automatic rollback.

Usage:
    API_URL=https://staging.churn-api.example.com pytest tests/smoke/ -v
"""

from __future__ import annotations

import os
import httpx
import pytest

API_URL = os.getenv("API_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=API_URL, timeout=15.0) as c:
        yield c


def test_health_check(client):
    """GET /health → 200 {status: healthy, model_version, model_loaded}."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert "model_version" in body
    assert body["model_loaded"] is True


def test_valid_predict(client):
    """POST /predict valid payload → 200 {churn_probability: 0..1}."""
    payload = {
        "tenure": 12,
        "MonthlyCharges": 55.0,
        "TotalCharges": 660.0,
        "SeniorCitizen": 0,
        "Contract": "One year",
        "PaymentMethod": "Mailed check",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "Churn": "No",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    prob = r.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_invalid_predict_returns_422(client):
    """POST /predict invalid payload → 422 (NOT 500 — slide 15)."""
    r = client.post("/predict", json={"tenure": "bad"})
    assert r.status_code == 422, (
        f"Expected 422 for invalid payload, got {r.status_code}. "
        f"A 500 means the server crashed, not just bad input."
    )
