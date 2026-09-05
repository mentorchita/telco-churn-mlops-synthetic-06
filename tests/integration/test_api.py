"""
Integration tests for the Telco Churn FastAPI /predict endpoint.
(Prompt 3 from slide 27 — 7 test cases)

These run against a LIVE staging API. Set API_URL env var or use default.

Usage in CI:
    API_URL=https://staging.churn-api.example.com pytest tests/integration/ \
        --junitxml=test-results/integration.xml
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 10.0   # seconds


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client() -> httpx.Client:
    """Reusable HTTP client for all integration tests."""
    with httpx.Client(base_url=API_URL, timeout=TIMEOUT) as c:
        yield c


@pytest.fixture(scope="session")
def sample_customer() -> dict:
    return {
        "tenure": 24,
        "MonthlyCharges": 65.5,
        "TotalCharges": 1572.0,
        "SeniorCitizen": 0,
        "Contract": "One year",
        "PaymentMethod": "Bank transfer (automatic)",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "TechSupport": "No",
        "Churn": "No",
    }


# ── L1: API Contract ──────────────────────────────────────────────────────────

class TestApiContract:
    """Slide 17 Level 1 — API Contract Tests."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_body_has_required_fields(self, client):
        response = client.get("/health")
        body = response.json()
        assert "model_version" in body
        assert "status" in body
        assert "model_loaded" in body

    def test_health_status_is_healthy(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "healthy"

    def test_predict_valid_payload_returns_200(self, client, sample_customer):
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 200

    def test_predict_response_has_churn_probability(self, client, sample_customer):
        response = client.post("/predict", json=sample_customer)
        body = response.json()
        assert "churn_probability" in body
        assert isinstance(body["churn_probability"], float)

    def test_predict_probability_in_range(self, client, sample_customer):
        response = client.post("/predict", json=sample_customer)
        prob = response.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0, f"Probability {prob} outside [0, 1]"

    def test_predict_missing_field_returns_422(self, client):
        """Invalid payload must return 422, NOT 500 (slide 15 smoke test)."""
        bad_payload = {"tenure": 12}   # many required fields missing
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422


# ── L2: Model Behaviour ───────────────────────────────────────────────────────

class TestModelBehaviour:
    """Slide 17 Level 2 — Model Behaviour Tests."""

    def test_high_risk_customer_has_high_probability(self, client, high_risk_customer):
        """tenure=1, Month-to-month → probability > 0.5 (slide 17)."""
        response = client.post("/predict", json=high_risk_customer)
        assert response.status_code == 200
        prob = response.json()["churn_probability"]
        assert prob > 0.5, (
            f"High-risk customer (tenure=1, M-t-M) got probability {prob:.3f}. "
            f"Expected > 0.5. Model logic may be inverted."
        )

    def test_low_risk_customer_has_low_probability(self, client, low_risk_customer):
        """tenure=60, Two year → probability < 0.4 (slide 17)."""
        response = client.post("/predict", json=low_risk_customer)
        assert response.status_code == 200
        prob = response.json()["churn_probability"]
        assert prob < 0.4, (
            f"Low-risk customer (tenure=60, Two year) got probability {prob:.3f}. "
            f"Expected < 0.4."
        )

    def test_churn_predicted_flag_consistent(self, client, sample_customer):
        """churn_predicted should be True iff churn_probability >= 0.5."""
        response = client.post("/predict", json=sample_customer)
        body = response.json()
        expected = body["churn_probability"] >= 0.5
        assert body["churn_predicted"] == expected


# ── L3: Performance SLA ───────────────────────────────────────────────────────

class TestPerformanceSLA:
    """Slide 17 Level 3 — Performance SLA Tests."""

    def test_single_predict_latency_under_200ms(self, client, sample_customer):
        """P99 latency < 200 ms (slide 17)."""
        # Run 10 requests, take the max
        latencies = []
        for _ in range(10):
            t0 = time.monotonic()
            response = client.post("/predict", json=sample_customer)
            latencies.append((time.monotonic() - t0) * 1000)
            assert response.status_code == 200
        if len(latencies) > 1:
            p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        else:
            p99 = latencies[-1]
        assert p99 < 200, f"P99 latency {p99:.1f} ms exceeds 200 ms threshold"

    def test_batch_50_customers_succeeds(self, client, sample_customer):
        """Batch of 50 customers → 200 + list of results (slide 27, Prompt 3)."""
        batch = {"customers": [sample_customer] * 50}
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 50
        for r in results:
            assert "churn_probability" in r
            assert 0.0 <= r["churn_probability"] <= 1.0
