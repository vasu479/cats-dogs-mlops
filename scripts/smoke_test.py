"""M4 - post-deployment smoke test.

Exercises the deployed service exactly as a user would: a health check followed
by a real prediction call. Exits with a non-zero status on any failure, which is
what makes the CD pipeline fail (see .github/workflows/cd.yml).

Usage:
    python scripts/smoke_test.py --base-url http://localhost:8000
    python scripts/smoke_test.py --base-url http://localhost:8000 --image path/to/cat.jpg
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LABELS = {"cat", "dog"}

PASS = "PASS"
FAIL = "FAIL"


def emit(status: str, name: str, detail: str = "") -> None:
    marker = "[ OK ]" if status == PASS else "[FAIL]"
    line = f"{marker} {name}"
    if detail:
        line += f" -> {detail}"
    print(line, flush=True)


def synthetic_jpeg() -> bytes:
    """Deterministic 224x224 JPEG so the test needs no fixture file on disk."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (224, 224), color=(200, 190, 170))
    draw = ImageDraw.Draw(image)
    for i in range(0, 224, 16):
        draw.line([(i, 0), (0, i)], fill=(90, 70, 60), width=3)
    draw.ellipse([60, 60, 164, 164], fill=(140, 110, 80), outline=(40, 30, 20), width=4)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def load_image(image_path: Optional[str]) -> Tuple[bytes, str]:
    if image_path:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"--image not found: {path}")
        return path.read_bytes(), path.name
    return synthetic_jpeg(), "synthetic_sample.jpg"


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def wait_for_health(base_url: str, timeout: int) -> dict:
    """Poll /health until the service reports a loaded model, or time out."""
    deadline = time.time() + timeout
    last_error = "no response"
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                body = response.json()
                if body.get("model_loaded") is True:
                    emit(PASS, "health check", f"attempt {attempt}: {body['status']}")
                    return body
                last_error = f"model_loaded=false ({body.get('error')})"
            else:
                last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = type(exc).__name__
        time.sleep(2)

    emit(FAIL, "health check", f"gave up after {timeout}s: {last_error}")
    raise SystemExit(1)


def check_prediction(base_url: str, image_bytes: bytes, filename: str) -> dict:
    try:
        response = requests.post(
            f"{base_url}/predict",
            files={"file": (filename, image_bytes, "image/jpeg")},
            timeout=30,
        )
    except requests.RequestException as exc:
        emit(FAIL, "prediction call", f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)

    if response.status_code != 200:
        emit(FAIL, "prediction call", f"HTTP {response.status_code}: {response.text[:200]}")
        raise SystemExit(1)

    body = response.json()

    for field in ("label", "confidence", "probabilities"):
        if field not in body:
            emit(FAIL, "prediction schema", f"missing field '{field}'")
            raise SystemExit(1)

    if body["label"] not in EXPECTED_LABELS:
        emit(FAIL, "prediction label", f"unexpected label {body['label']!r}")
        raise SystemExit(1)

    probability_sum = sum(body["probabilities"].values())
    if abs(probability_sum - 1.0) > 1e-2:
        emit(FAIL, "probability distribution", f"sums to {probability_sum:.4f}, not 1.0")
        raise SystemExit(1)

    if not 0.0 <= body["confidence"] <= 1.0:
        emit(FAIL, "confidence range", f"confidence={body['confidence']}")
        raise SystemExit(1)

    emit(
        PASS,
        "prediction call",
        f"label={body['label']} confidence={body['confidence']:.4f} "
        f"latency={body.get('inference_time_ms')}ms",
    )
    return body


def check_invalid_input_is_rejected(base_url: str) -> None:
    """A non-image upload must produce 400, not 500 - the service must not crash."""
    response = requests.post(
        f"{base_url}/predict",
        files={"file": ("junk.txt", b"not an image at all", "text/plain")},
        timeout=15,
    )
    if response.status_code != 400:
        emit(FAIL, "invalid input handling", f"expected HTTP 400, got {response.status_code}")
        raise SystemExit(1)
    emit(PASS, "invalid input handling", "non-image upload correctly rejected with 400")


def check_metrics(base_url: str) -> None:
    response = requests.get(f"{base_url}/metrics", timeout=10)
    if response.status_code != 200:
        emit(FAIL, "metrics endpoint", f"HTTP {response.status_code}")
        raise SystemExit(1)
    for metric in ("app_requests_total", "app_request_latency_ms"):
        if metric not in response.text:
            emit(FAIL, "metrics endpoint", f"missing metric '{metric}'")
            raise SystemExit(1)
    emit(PASS, "metrics endpoint", "request counters and latency summary exposed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-deploy smoke test.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--image", default=None, help="Optional real image to send.")
    parser.add_argument(
        "--timeout", type=int, default=60, help="Seconds to wait for /health."
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"\n=== SMOKE TEST :: {base_url} ===\n", flush=True)

    started = time.time()
    health = wait_for_health(base_url, args.timeout)
    image_bytes, filename = load_image(args.image)
    prediction = check_prediction(base_url, image_bytes, filename)
    check_invalid_input_is_rejected(base_url)
    check_metrics(base_url)

    elapsed = time.time() - started
    print(
        "\n"
        + json.dumps(
            {
                "result": "PASSED",
                "base_url": base_url,
                "elapsed_seconds": round(elapsed, 2),
                "service_version": health.get("version"),
                "git_sha": health.get("git_sha"),
                "sample_prediction": {
                    "label": prediction["label"],
                    "confidence": prediction["confidence"],
                },
            },
            indent=2,
        ),
        flush=True,
    )
    print("\n=== ALL SMOKE TESTS PASSED ===\n", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        emit(FAIL, "smoke test", f"unhandled {type(exc).__name__}: {exc}")
        sys.exit(1)
