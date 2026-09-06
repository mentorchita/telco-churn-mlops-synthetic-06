"""
FastAPI serving endpoint for Telco Churn prediction.

Endpoints:
    GET  /health   → health check with model version
    POST /predict  → single-customer churn probability
    POST /predict/batch → up to 200 customers

Environment variables:
    MODEL_URI      MLflow model URI, e.g. models:/telco-churn-prod/Production
    MLFLOW_TRACKING_URI  MLflow server URL

Slide 15 (staging smoke test):
    GET /health → 200 {status, model_version, model_loaded}
    POST /predict valid   → 200 {churn_probability: 0..1}
    POST /predict invalid → 422

Slide 17 (integration tests L1-L4): all test cases hit these endpoints.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMN, encode_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telco Churn Prediction API",
    description="XGBoost churn predictor — Modern MLOps Module 6",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state ────────────────────────────────────────────────────────
_model = None
_model_version: str = "unknown"
_model_loaded: bool = False
_load_time_s: float = 0.0


def load_model() -> None:
    """Load model from MLflow when available, else local joblib fallback."""
    global _model, _model_version, _model_loaded, _load_time_s

    t0 = time.time()
    model_uri = os.getenv("MODEL_URI", "models:/telco-churn-prod/Production").strip()
    model_path = os.getenv("MODEL_PATH", "models/churn_model.joblib")

    # Try MLflow first (real deployments with tracking server / registry)
    try:
        logger.info("Loading model from MLflow URI %s", model_uri)
        _model = mlflow.pyfunc.load_model(model_uri)
        _model_version = os.getenv("MODEL_VERSION", model_uri.split("/")[-1])
        _model_loaded = True
        _load_time_s = round(time.time() - t0, 2)
        logger.info("Model loaded from MLflow in %.2f s", _load_time_s)
        return
    except Exception as exc:
        logger.warning("MLflow load failed (%s); trying local fallback", exc)

    # Local joblib fallback (CI / image without MLflow server)
    try:
        import joblib

        logger.info("Loading local model from %s", model_path)
        _model = joblib.load(model_path)
        _model_version = os.getenv("MODEL_VERSION", "local-fallback")
        _model_loaded = True
        _load_time_s = round(time.time() - t0, 2)
        logger.info("Local model loaded in %.2f s  version=%s", _load_time_s, _model_version)
    except Exception as exc:
        logger.error("Failed to load any model: %s", exc)
        _model_loaded = False


@app.on_event("startup")
async def startup_event() -> None:
    load_model()


# ── Schemas ───────────────────────────────────────────────────────────────────


class CustomerInput(BaseModel):
    """Single customer for churn prediction.

    Field constraints mirror the Pandera schema in tests/unit/test_schema.py
    so that validation is consistent between training-time and serving-time.
    """

    customerID: Optional[str] = None
    tenure: int = Field(..., ge=0, le=72, description="Months as customer")
    MonthlyCharges: float = Field(..., ge=0, le=300)
    TotalCharges: Optional[float] = Field(None, ge=0)
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Contract: str = Field(..., description="Month-to-month | One year | Two year")
    PaymentMethod: str
    InternetService: str
    OnlineSecurity: str
    TechSupport: str
    Churn: Optional[str] = "No"  # ignored in prediction, required by encode_features

    @validator("Contract")
    def validate_contract(cls, v):
        allowed = {"Month-to-month", "One year", "Two year"}
        if v not in allowed:
            raise ValueError(f"Contract must be one of {allowed}")
        return v

    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    customerID: Optional[str] = None
    churn_probability: float
    churn_predicted: bool
    model_version: str


class BatchRequest(BaseModel):
    customers: List[CustomerInput] = Field(..., max_items=200)


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
    load_time_s: float


# ── Helpers ───────────────────────────────────────────────────────────────────


def _predict_single(customer: CustomerInput) -> float:
    """Run model inference, return churn probability."""
    if not _model_loaded or _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([customer.dict()])
    encoded = encode_features(row)
    X_cols = [c for c in FEATURE_COLUMNS if c != TARGET_COLUMN]
    X = encoded[X_cols].values

    # sklearn/xgboost: prefer predict_proba; mlflow pyfunc: predict
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)
        if getattr(proba, "ndim", 1) == 2:
            return float(proba[0, 1])
        return float(proba[0])
    proba = _model.predict(X)
    if getattr(proba, "ndim", 1) == 2:
        return float(proba[0, 1])
    return float(proba[0])


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Health check — used in staging smoke test (slide 15)."""
    return HealthResponse(
        status="healthy" if _model_loaded else "degraded",
        model_version=_model_version,
        model_loaded=_model_loaded,
        load_time_s=_load_time_s,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(customer: CustomerInput) -> PredictionResponse:
    """Predict churn probability for a single customer."""
    prob = _predict_single(customer)
    return PredictionResponse(
        customerID=customer.customerID,
        churn_probability=round(prob, 4),
        churn_predicted=prob >= 0.5,
        model_version=_model_version,
    )


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["prediction"])
def predict_batch(request: BatchRequest) -> List[PredictionResponse]:
    """Predict churn probability for up to 200 customers at once."""
    results = []
    for customer in request.customers:
        prob = _predict_single(customer)
        results.append(
            PredictionResponse(
                customerID=customer.customerID,
                churn_probability=round(prob, 4),
                churn_predicted=prob >= 0.5,
                model_version=_model_version,
            )
        )
    return results


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Telco Churn API — see /docs for Swagger UI"}
