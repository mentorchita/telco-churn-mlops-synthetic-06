# ── Stage 1: install Python deps ──────────────────────────────────────────────
# This layer is cached by Docker BuildKit.  Re-built only when requirements.txt changes.
# Slide 13: multi-stage build → ~450 MB final image vs ~3 GB naïve build.
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# ── Stage 2: lean production image ────────────────────────────────────────────
FROM python:3.11-slim

# Security: run as non-root user (slide 13)
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy only the installed packages from Stage 1 (no dev tools in final image)
COPY --from=builder /root/.local /home/appuser/.local

# Copy application source
COPY src/ ./src/

# Python path for module imports
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

# MLflow model URI — loaded at container startup, NOT baked into image.
# Update this env var to deploy a new model version without rebuilding.
# Slide 13: "Model loaded from MLflow URI, not baked into image."
ENV MODEL_URI=models:/telco-churn-prod/Production
ENV MODEL_VERSION=Production

# Optionally set tracking server
# ENV MLFLOW_TRACKING_URI=http://mlflow:5000

USER appuser

EXPOSE 8000

# Used for liveness probe in staging smoke test (slide 15) and Kubernetes (Module 7)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
