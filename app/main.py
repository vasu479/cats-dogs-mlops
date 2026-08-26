"""M2 + M5 - FastAPI inference service.

Endpoints
---------
GET  /            service banner
GET  /health      liveness + readiness (M2 required endpoint #1)
GET  /ready       strict readiness probe (503 until the model is loaded)
POST /predict     multipart image -> class label + probabilities (M2 endpoint #2)
GET  /metrics     Prometheus exposition text (M5)
GET  /metrics/json  the same counters as JSON, easier to show in a demo (M5)
GET  /model-info  metadata about the loaded checkpoint

Logging (M5): every request/response pair is logged as one structured JSON line
containing a request id, endpoint, status, latency and - for predictions - the
predicted label and confidence. Raw image bytes and the uploaded filename's
content are NEVER logged; only the size and a short content hash are recorded,
so the log is safe to ship off-box.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from hashlib import sha256
from typing import Any, Dict

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from src.config import CONFIG
from src.inference import InvalidImageError, ModelService
from src.monitoring import METRICS

# ---------------------------------------------------------------------------
# Structured logging: one JSON object per line on stdout, which is what
# `docker compose logs` and any log shipper expect.
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
LOGGER = logging.getLogger("cats-dogs-api")
LOGGER.setLevel(LOG_LEVEL)
LOGGER.handlers = [_handler]
LOGGER.propagate = False


def log_event(**fields: Any) -> None:
    """Emit one structured log line. Never called with raw image bytes."""
    fields.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    LOGGER.info(json.dumps(fields, default=str))


MODEL_PATH = os.getenv("MODEL_PATH", str(CONFIG.serving.model_path))
SERVICE = ModelService(MODEL_PATH, CONFIG.serving.class_names)
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
GIT_SHA = os.getenv("GIT_SHA", "local")


@asynccontextmanager
async def lifespan(_: FastAPI):
    loaded = SERVICE.load()
    log_event(
        event="startup",
        model_path=MODEL_PATH,
        model_loaded=loaded,
        error=SERVICE.load_error,
        version=APP_VERSION,
        git_sha=GIT_SHA,
    )
    yield
    log_event(event="shutdown")


app = FastAPI(
    title="Cats vs Dogs Classifier",
    description=(
        "Binary image classification service for a pet adoption platform. "
        "MLOps Assignment 2 (AIMLCZG523)."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """M5: request/response logging + request count and latency metrics."""
    request_id = str(uuid.uuid4())
    endpoint = request.url.path
    started = time.perf_counter()

    METRICS.record_request(endpoint)
    log_event(
        event="request",
        request_id=request_id,
        method=request.method,
        endpoint=endpoint,
        client=request.client.host if request.client else None,
    )

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - started) * 1000
        METRICS.record_error(type(exc).__name__)
        METRICS.record_response(endpoint, 500, latency_ms)
        log_event(
            event="response",
            request_id=request_id,
            endpoint=endpoint,
            status=500,
            latency_ms=round(latency_ms, 2),
            error=type(exc).__name__,
        )
        raise

    latency_ms = (time.perf_counter() - started) * 1000
    METRICS.record_response(endpoint, status_code, latency_ms)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{latency_ms:.2f}"
    log_event(
        event="response",
        request_id=request_id,
        endpoint=endpoint,
        status=status_code,
        latency_ms=round(latency_ms, 2),
    )
    return response


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "cats-vs-dogs-classifier",
        "version": APP_VERSION,
        "git_sha": GIT_SHA,
        "endpoints": ["/health", "/ready", "/predict", "/metrics", "/model-info", "/docs"],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness + readiness. Always 200 so a container is not killed while the
    model is still loading; ``model_loaded`` carries the readiness signal."""
    return {
        "status": "healthy" if SERVICE.is_ready else "degraded",
        "model_loaded": SERVICE.is_ready,
        "model_path": str(SERVICE.model_path),
        "version": APP_VERSION,
        "git_sha": GIT_SHA,
        "error": SERVICE.load_error,
    }


@app.get("/ready")
def ready() -> JSONResponse:
    """Strict readiness probe: 503 until the checkpoint is loaded."""
    if SERVICE.is_ready:
        return JSONResponse(status_code=200, content={"ready": True})
    return JSONResponse(
        status_code=503, content={"ready": False, "error": SERVICE.load_error}
    )


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    return {"loaded": SERVICE.is_ready, "metadata": SERVICE.metadata}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """Classify one uploaded image as cat or dog.

    Returns the predicted label, its confidence and the full probability
    distribution over both classes.
    """
    content = await file.read()

    # Privacy: log only size and a truncated content hash - never the bytes.
    content_hash = sha256(content).hexdigest()[:12] if content else "empty"

    if not SERVICE.is_ready:
        METRICS.record_error("ModelNotLoaded")
        log_event(
            event="prediction_rejected",
            reason="model_not_loaded",
            content_sha256_12=content_hash,
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model is not loaded.",
                "detail": SERVICE.load_error,
                "hint": "Train the model (python -m src.train) or mount models/model.pt.",
            },
        )

    try:
        payload, inference_ms = SERVICE.predict(content)
    except InvalidImageError as exc:
        METRICS.record_error("InvalidImage")
        log_event(
            event="prediction_rejected",
            reason="invalid_image",
            detail=str(exc),
            bytes=len(content),
            content_sha256_12=content_hash,
        )
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:  # noqa: BLE001
        METRICS.record_error(type(exc).__name__)
        log_event(event="prediction_failed", error=type(exc).__name__, detail=str(exc))
        return JSONResponse(
            status_code=500, content={"error": "Inference failed.", "detail": str(exc)}
        )

    METRICS.record_prediction(str(payload["label"]))
    log_event(
        event="prediction",
        content_type=file.content_type,
        bytes=len(content),
        content_sha256_12=content_hash,
        label=payload["label"],
        confidence=payload["confidence"],
        inference_ms=inference_ms,
    )
    return JSONResponse(
        status_code=200, content={**payload, "inference_time_ms": inference_ms}
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """M5: Prometheus exposition format."""
    return METRICS.prometheus_text()


@app.get("/metrics/json")
def metrics_json() -> Dict[str, Any]:
    """M5: the same counters as JSON - easier to read in a screen recording."""
    return METRICS.snapshot()
