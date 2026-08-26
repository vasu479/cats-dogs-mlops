"""M3 - integration tests for the FastAPI service.

These run against the in-process ASGI app, so no container is required. They
pass both with and without a trained checkpoint present: when the model is
absent the service must degrade cleanly (503 with a useful message) rather
than crash, which is itself the behaviour worth asserting.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (300, 300), color=(180, 140, 90)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_endpoint_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] in {"healthy", "degraded"}
    assert "model_loaded" in body
    assert "version" in body


def test_root_lists_endpoints(client: TestClient) -> None:
    body = client.get("/").json()
    assert "/health" in body["endpoints"]
    assert "/predict" in body["endpoints"]


def test_predict_returns_label_and_probabilities(
    client: TestClient, jpeg_bytes: bytes
) -> None:
    model_loaded = client.get("/health").json()["model_loaded"]
    response = client.post(
        "/predict", files={"file": ("sample.jpg", jpeg_bytes, "image/jpeg")}
    )

    if not model_loaded:
        # No checkpoint in this environment: the service must say so, not 500.
        assert response.status_code == 503
        assert "error" in response.json()
        pytest.skip("No trained model available; asserted graceful degradation.")

    assert response.status_code == 200
    body = response.json()

    assert body["label"] in {"cat", "dog"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"]) == {"cat", "dog"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert isinstance(body["inference_time_ms"], int)


def test_predict_rejects_a_non_image_upload(client: TestClient) -> None:
    model_loaded = client.get("/health").json()["model_loaded"]
    response = client.post(
        "/predict", files={"file": ("notes.txt", b"plain text", "text/plain")}
    )
    assert response.status_code == (400 if model_loaded else 503)


def test_predict_requires_a_file(client: TestClient) -> None:
    assert client.post("/predict").status_code == 422


def test_metrics_endpoint_exposes_prometheus_counters(client: TestClient) -> None:
    client.get("/health")  # generate at least one request
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "app_requests_total" in response.text
    assert "app_request_latency_ms" in response.text


def test_metrics_json_counts_requests_and_latency(client: TestClient) -> None:
    client.get("/health")
    snapshot = client.get("/metrics/json").json()

    assert snapshot["requests_total"] > 0
    assert "/health" in snapshot["requests_by_endpoint"]
    assert snapshot["latency_ms"]["count"] > 0


def test_response_carries_request_id_and_timing_headers(client: TestClient) -> None:
    response = client.get("/health")
    assert response.headers.get("X-Request-ID")
    assert float(response.headers["X-Process-Time-ms"]) >= 0
