"""
FastAPI model-serving application for the heart disease classifier.

Endpoints:
  GET  /health   - liveness/readiness probe
  POST /predict  - returns prediction + confidence for a single patient
  GET  /metrics  - Prometheus metrics (via prometheus-fastapi-instrumentator)

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import PatientFeatures, PredictionResponse

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"
METADATA_PATH = ROOT / "models" / "metadata.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("heart-disease-api")

_model = None
_model_version = "unknown"


def get_model():
    global _model, _model_version
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model file not found at {MODEL_PATH}. Run `python -m src.train` first."
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        if METADATA_PATH.exists():
            meta = json.loads(METADATA_PATH.read_text())
            _model_version = meta.get("best_model", "unknown")
        logger.info("Model loaded: version=%s", _model_version)
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_model()
    except RuntimeError as exc:
        logger.warning("Model not loaded at startup: %s", exc)
    yield


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts heart disease risk from patient health data.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s -> %s (%.2fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "degraded", "model_version": _model_version}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    try:
        model = get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    df = pd.DataFrame([features.model_dump()])
    try:
        pred = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][pred])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    logger.info("Prediction made: pred=%s confidence=%.4f", pred, proba)
    return PredictionResponse(
        prediction=pred,
        label="Disease" if pred == 1 else "No Disease",
        confidence=round(proba, 4),
        model_version=_model_version,
    )
