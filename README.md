# Telco Churn MLOps — Synthetic Dataset

**Modern MLOps · LLMOps · Agentic Ops — Module 6 demo project**

XGBoost churn prediction model with a full MLOps CI/CD pipeline.
Used as the hands-on example throughout Module 6: CI/CD Pipelines.

---

## Repository Structure

```
telco-churn-mlops-synthetic-05/
├── src/
│   ├── features/build_features.py   ← feature engineering (encode_features)
│   ├── models/train.py              ← XGBoost training + MLflow + quality gate
│   └── api/main.py                  ← FastAPI serving (/health, /predict)
├── tests/
│   ├── conftest.py                  ← shared fixtures (sample_raw_df, etc.)
│   ├── unit/
│   │   ├── test_features.py         ← unit tests for encode_features()
│   │   ├── test_model.py            ← model quality gate (AUC/F1/Precision)
│   │   ├── test_schema.py           ← Pandera schema validation
│   │   └── test_drift.py            ← scheduled drift detection
│   ├── integration/
│   │   └── test_api.py              ← FastAPI integration tests (L1–L3)
│   └── smoke/
│       └── test_smoke.py            ← post-deploy smoke test (30 s)
├── .github/workflows/
│   ├── ml-pipeline.yml              ← full CI/CD pipeline (5 jobs)
│   └── drift-check.yml              ← daily drift detection (cron 03:00)
├── Dockerfile                       ← multi-stage build (~450 MB)
├── .dockerignore
├── locustfile.py                    ← load test (50 users, p95/p99/error_rate)
├── requirements.txt                 ← production deps
├── requirements-dev.txt             ← dev/CI deps (flake8, black, mypy, pytest)
├── pyproject.toml                   ← tool config (black, mypy, coverage ≥80%)
└── pytest.ini                       ← pytest markers (slow, integration, smoke)
```

---

## Quality Gate Thresholds (Slide 12)

| Metric | Threshold | Why |
|--------|-----------|-----|
| ROC-AUC | ≥ 0.82 | 34% better than DummyClassifier |
| F1-Score | ≥ 0.70 | Balances precision/recall |
| Precision (cls=1) | ≥ 0.75 | Controls retention campaign waste |
| Feature count | = 19 | Catches silent feature changes |

If any threshold fails → `sys.exit(1)` → pipeline stops before Docker build.

---

## CI/CD Pipeline (Slide 5)

```
Code Push / PR
    ↓
lint-and-test      (flake8 · black · mypy · pandera · pytest unit · pytest model)
    ↓  needs: lint-and-test
build-and-push     (multi-stage Docker · GHCR tags: sha + branch + semver)
    ↓  needs: build-and-push
deploy-staging     (docker run on staging VM · 20 s wait · /health check)
    ↓  needs: deploy-staging
integration-tests  (smoke test · API contract · model behaviour · latency SLA)
    ↓  needs: integration-tests  +  manual approval
deploy-production  (docker run on prod VM · Slack notification)
```

Daily at 03:00 UTC → `drift-check.yml` runs independently (no code push needed).

---

## Quick Start (Lab 1)

```bash
# 1. Fork this repo → clone locally
git clone https://github.com/YOUR-FORK/telco-churn-mlops-synthetic-05
cd telco-churn-mlops-synthetic-05

# 2. Install dev dependencies
pip install -r requirements-dev.txt

# 3. Run unit tests locally (no model training)
pytest tests/unit/test_features.py tests/unit/test_schema.py -v

# 4. Run model quality gate (takes 2-5 min)
pytest tests/unit/test_model.py -v -m slow

# 5. Run all unit tests with coverage
pytest tests/unit/ --cov=src --cov-report=html --cov-fail-under=80

# 6. Code quality checks
flake8 src/ tests/ --max-line-length 100
black --check src/ tests/
mypy src/ --ignore-missing-imports
```

---

## Lab 2 — Break CI → Debug → Fix

### Scenario A: AUC Threshold
```python
# In tests/unit/test_model.py — change to trigger CI failure:
MIN_AUC = 0.99   # was 0.82

# Symptom: AssertionError: 0.887 < 0.99
# Fix: restore MIN_AUC = 0.82
```

### Scenario B: Missing Import
```python
# In src/features/build_features.py — add at top:
import nonexistent_lib   # triggers ModuleNotFoundError

# Symptom: ModuleNotFoundError: No module named 'nonexistent_lib'
# Fix: remove the line, ensure all deps are in requirements.txt
```

### Scenario C: Data Type Mismatch
```python
# In tests/conftest.py — change tenure dtype:
"tenure": [1, 24, 60, 12, 6],          # correct: int
"tenure": ["1", "24", "60", "12", "6"], # break: str

# Symptom: pandera.errors.SchemaError: Expected int64, got object
# Fix: restore int dtype
```

---

## Lab 3 — Rollback

```bash
# 1. Deploy new image
docker stop churn-api || true
docker run -d -p 8000:8000 --name churn-api \
  -e MODEL_URI=models:/telco-churn-prod/Production \
  ghcr.io/YOUR-ORG/churn-api:sha-NEW

# 2. Smoke test
curl -f http://localhost:8000/health

# 3. Rollback (Code Rollback — slide 22)
docker stop churn-api
docker run -d -p 8000:8000 --name churn-api \
  -e MODEL_URI=models:/telco-churn-prod/Production \
  ghcr.io/YOUR-ORG/churn-api:sha-PREV

# 4. Model rollback (without touching code — slide 22)
mlflow models transition-state \
  --model telco-churn-prod --version 3 --stage Archived
mlflow models transition-state \
  --model telco-churn-prod --version 2 --stage Production
```

---

## GitHub Secrets Required

| Secret | Where | Used in |
|--------|-------|---------|
| `GHCR_TOKEN` | Repository | Docker push to GHCR |
| `STAGING_HOST` | Environment: staging | SSH deploy |
| `STAGING_SSH_KEY` | Environment: staging | SSH deploy |
| `PROD_HOST` | Environment: production | SSH deploy |
| `PROD_SSH_KEY` | Environment: production | SSH deploy |
| `MLFLOW_TRACKING_URI` | Repository | Model loading + logging |
| `SLACK_WEBHOOK` | Repository | Deploy + drift notifications |

Create environments at: Settings → Environments → New environment

Set `production` environment to require **1 of 3 reviewers** (slide 18).

---

## Module Navigation

| Module | Topic |
|--------|-------|
| Module 5 | Containerization & Microservices ← built the Dockerfile |
| **Module 6** | **CI/CD Pipelines ← this repo** |
| Module 7 | Kubernetes Deployment → will use images built here |
| Module 8 | Monitoring & Observability |
