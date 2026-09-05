"""
Locust load test for the Telco Churn API.
(Prompt 5 from slide 28)

Usage:
    # Headless (CI):
    locust --headless -u 50 -r 10 --run-time 5m \
           --host http://staging:8000 \
           --html locust_report.html \
           --csv locust_stats \
           --exit-code-on-error 1

    # Interactive (local):
    locust --host http://localhost:8000

Fail criteria (slide 28):
    - p95 > 200 ms
    - p99 > 500 ms
    - error_rate > 1%
"""

from __future__ import annotations

import json
import random
from locust import HttpUser, TaskSet, between, events, task

# ── Sample customers ──────────────────────────────────────────────────────────

CUSTOMERS = [
    {
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
    },
    {
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
    },
    {
        "tenure": 24,
        "MonthlyCharges": 65.5,
        "TotalCharges": 1572.0,
        "SeniorCitizen": 1,
        "Contract": "One year",
        "PaymentMethod": "Mailed check",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "Churn": "No",
    },
]


# ── Users ─────────────────────────────────────────────────────────────────────

class ChurnApiUser(HttpUser):
    """Standard user — single /predict requests."""

    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        """Health check before starting load — stop if service is down."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")
                self.environment.runner.quit()
            else:
                body = response.json()
                if not body.get("model_loaded"):
                    response.failure("Model not loaded")
                    self.environment.runner.quit()

    @task(3)
    def predict_single(self) -> None:
        """POST /predict with a random customer profile."""
        customer = random.choice(CUSTOMERS)
        with self.client.post(
            "/predict",
            json=customer,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                prob = response.json().get("churn_probability", -1)
                if not (0.0 <= prob <= 1.0):
                    response.failure(f"Probability out of range: {prob}")
            elif response.status_code == 422:
                response.failure(f"Validation error: {response.text}")
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(1)
    def health_check(self) -> None:
        """Periodic health check to catch degradation under load."""
        self.client.get("/health")


class BatchUser(HttpUser):
    """Batch user — tests /predict/batch with 50 customers per request."""

    wait_time = between(1.0, 2.0)

    def on_start(self) -> None:
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code != 200:
                self.environment.runner.quit()

    @task
    def predict_batch(self) -> None:
        """POST /predict/batch with 50 customers."""
        batch_size = 50
        customers = [random.choice(CUSTOMERS) for _ in range(batch_size)]
        with self.client.post(
            "/predict/batch",
            json={"customers": customers},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                results = response.json()
                if len(results) != batch_size:
                    response.failure(f"Expected {batch_size} results, got {len(results)}")
            else:
                response.failure(f"Batch predict failed: {response.status_code}")


# ── Threshold validation at test end ─────────────────────────────────────────

@events.quitting.add_listener
def check_thresholds(environment, **kwargs) -> None:
    """Fail the test if any SLA threshold is exceeded (slide 28)."""
    stats = environment.stats.total
    failures = []

    if stats.num_requests == 0:
        print("No requests completed — cannot check thresholds")
        return

    p95 = stats.get_response_time_percentile(0.95)
    p99 = stats.get_response_time_percentile(0.99)
    error_rate = stats.fail_ratio * 100

    if p95 and p95 > 200:
        failures.append(f"p95 latency {p95:.0f} ms > 200 ms threshold")
    if p99 and p99 > 500:
        failures.append(f"p99 latency {p99:.0f} ms > 500 ms threshold")
    if error_rate > 1.0:
        failures.append(f"Error rate {error_rate:.1f}% > 1% threshold")

    if failures:
        print("\n⚠  SLA THRESHOLDS EXCEEDED:")
        for f in failures:
            print(f"  ✗ {f}")
        environment.process_exit_code = 1
    else:
        print(f"\n✓ All SLA thresholds met: p95={p95:.0f}ms, p99={p99:.0f}ms, errors={error_rate:.1f}%")
