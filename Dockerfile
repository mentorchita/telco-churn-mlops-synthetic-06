# ── Stage 1: install deps + train fallback model ─────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Train a small local model so /health is healthy without MLflow in CI
COPY src/ ./src/
COPY scripts/build_fallback_model.py ./scripts/build_fallback_model.py
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
RUN python scripts/build_fallback_model.py


# ── Stage 2: lean production image ────────────────────────────────────────────
FROM python:3.11-slim

RUN useradd --create-home --shell /bin/bash appuser \
    && apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local
COPY --from=builder /app/models /app/models
COPY src/ ./src/

ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

# Prefer MLflow when configured; otherwise local joblib fallback
ENV MODEL_URI=models:/telco-churn-prod/Production
ENV MODEL_PATH=/app/models/churn_model.joblib
ENV MODEL_VERSION=fallback

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
