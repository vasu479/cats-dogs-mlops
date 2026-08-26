# Assignment 2 — Rubric Mapping and Evidence

**Course:** MLOps (S1-25_AIMLCZG523) · **Total marks:** 50
**Student:** Sreenivasulu Remuri · **BITS ID:** 2024AC05343
**Use case:** Binary image classification (cats vs dogs) for a pet adoption platform
**Dataset:** Kaggle Cats-and-Dogs, pre-processed to 224×224 RGB, split 80/10/10, augmented

This document maps every task in the assignment brief to the file that implements it and the
artifact that proves it.

---

## M1 — Model Development & Experiment Tracking (10 marks)

### 1. Data & Code Versioning

| Requirement | Implementation | Evidence |
| --- | --- | --- |
| Git for source code versioning | Repository with `src/`, `app/`, `tests/`, `scripts/`, notebooks-free structure; `.gitattributes` enforces LF | `git log`, repo tree |
| DVC for dataset versioning | `dvc add data/raw` → `data/raw.dvc`; local DVC remote configured by `setup_windows.ps1` | `data/raw.dvc`, `.dvc/config` |
| DVC tracks pre-processed data | `dvc.yaml` stage **prepare** declares `data/processed` as an output | `dvc.yaml`, `dvc.lock`, `dvc dag` |

The DVC pipeline is reproducible end to end: `dvc repro` re-runs only the stages whose
dependencies or parameters changed.

### 2. Model Building

| Requirement | Implementation | Evidence |
| --- | --- | --- |
| Baseline model (simple CNN) | `src/model.py` — `SimpleCNN`: 4 × (Conv-BN-ReLU-MaxPool) → global average pool → dropout → linear | `src/model.py` |
| 224×224 RGB pre-processing | `src/data_prep.py::preprocess_image` guarantees RGB mode and exact 224×224 size | `data/processed/summary.json` |
| 80/10/10 split | `src/data_prep.py::stratified_split` — deterministic, exhaustive, disjoint, seeded | `data/processed/manifest.csv` |
| Data augmentation | `src/datasets.py::build_train_transform` — RandomResizedCrop, HorizontalFlip, Rotation(15°), ColorJitter. Train split only. | `src/datasets.py` |
| Serialized model | `torch.save` → `models/model.pt`, a checkpoint dict carrying weights, class names, image size and training metrics | `models/model.pt` |

### 3. Experiment Tracking

MLflow (`file:./mlruns`), experiment `cats-vs-dogs`.

| Logged | Detail |
| --- | --- |
| Parameters | model name, epochs, batch size, learning rate, weight decay, optimiser, image size, all three split ratios, augmentation description, seed, parameter counts, split sizes |
| Metrics (per epoch) | `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy` |
| Metrics (final) | `test_accuracy`, `test_precision`, `test_recall`, `test_f1`, `best_val_accuracy`, `training_seconds` |
| Artifacts | `confusion_matrix.png`, `loss_curves.png`, `classification_report.txt`, `metrics.json`, `model.pt` |

**Screenshot for submission:** `mlflow ui` → the run page showing parameters, the metric
curves, and the Artifacts tab with the confusion matrix and loss curves.

---

## M2 — Model Packaging & Containerization (10 marks)

### 1. Inference Service

`app/main.py` (FastAPI). Two required endpoints, plus extras:

| Endpoint | Requirement | Behaviour |
| --- | --- | --- |
| `GET /health` | **health check** | 200 with `status`, `model_loaded`, `version`, `git_sha` |
| `POST /predict` | **prediction** | multipart image → `label`, `label_index`, `confidence`, `probabilities` (both classes), `inference_time_ms` |
| `GET /ready` | extra | 503 until the checkpoint loads — a real readiness probe |
| `GET /model-info` | extra | checkpoint metadata and training metrics |

Input validation: empty uploads, non-image payloads and files over 10 MB return **400**, not
500. The service degrades to **503** with a diagnostic message if the model is absent, rather
than crashing on startup.

### 2. Environment Specification

`requirements.txt` — runtime only, every dependency pinned to an exact version
(`torch==2.8.0`, `torchvision==0.23.0`, `numpy==2.1.3`, `fastapi==0.115.6`,
`pillow==11.0.0`, …). `requirements-dev.txt` adds the pinned training and test stack
(`mlflow==2.20.3`, `scikit-learn==1.6.1`, `dvc==3.59.1`, `pytest==8.3.4`, …).

MLflow, scikit-learn and matplotlib are deliberately excluded from the runtime image: they are
training-time only. The runtime import graph was verified to require nothing outside
`requirements.txt`.

### 3. Containerization

`Dockerfile` — multi-stage build:

- **builder** stage resolves dependencies into `/opt/venv`, using
  `--extra-index-url https://download.pytorch.org/whl/cpu` so torch resolves to the ~200 MB
  CPU wheel rather than the ~2.5 GB CUDA build;
- **runtime** stage is `python:3.11-slim` with only the venv, `src/`, `app/`, `params.yaml`
  and `models/` copied in;
- runs as unprivileged user `appuser` (uid 10001);
- stdlib-only `HEALTHCHECK` (no curl in the image);
- `GIT_SHA` build arg so `/health` reports the exact commit serving traffic.

**Verification for submission:** `docker compose up -d` then
`curl.exe http://localhost:8000/health` and
`curl.exe -X POST http://localhost:8000/predict -F "file=@<image>.jpg"`.

---

## M3 — CI Pipeline for Build, Test & Image Creation (10 marks)

### 1. Automated Testing — `pytest`, 44 tests

| File | Covers | Count |
| --- | --- | --- |
| `tests/test_data_prep.py` | **data pre-processing function** — `preprocess_image` (RGB/size guarantees across grayscale, RGBA, palette, portrait, landscape inputs; invalid-size rejection) and `stratified_split` (exact ratios, exhaustive, disjoint, deterministic, order-insensitive, ratio validation) | 19 |
| `tests/test_model_utils.py` | **model/inference utility** — `bytes_to_tensor` (shape, dtype, normalisation, empty/corrupt/oversized rejection), `probabilities_to_response` (argmax, full distribution, class-count mismatch), model factory and checkpoint round-trip | 17 |
| `tests/test_api.py` | endpoint contracts, error paths, metrics, response headers | 8 |

The assignment requires at least one pre-processing test and one model-utility test; both
categories are covered several times over.

### 2. CI Setup — GitHub Actions (`.github/workflows/ci.yml`)

Triggers on every push to `main`/`develop`, every pull request to `main`, and manual dispatch.

Job **test**: checkout → set up Python 3.11 (pip cache) → install `requirements-dev.txt` →
`pytest --cov` → upload the coverage report.

Job **build-and-push** (needs `test`, so a red test suite blocks the build): checkout →
Buildx → GHCR login → verify `models/model.pt` exists → build image → start the container and
assert `/health` reports `model_loaded: true` → tear down.

### 3. Artifact Publishing

Pushes to **GitHub Container Registry**:

```
ghcr.io/vasu479/cats-dogs-api:latest
ghcr.io/vasu479/cats-dogs-api:sha-<7-char-commit>
```

Authenticated with the per-run `GITHUB_TOKEN` — no long-lived secret is stored in the
repository. Pull requests build but do **not** push.

---

## M4 — CD Pipeline & Deployment (10 marks)

### 1. Deployment Target

**Docker Compose.** `docker-compose.yml` defines the service, port mapping, environment,
container healthcheck, restart policy, project labels and a dedicated bridge network. The same
manifest is used by the CD job and by the local demo, so CI proves exactly what is demonstrated.

### 2. CD / GitOps Flow — `.github/workflows/cd.yml`

Triggered by `workflow_run` when **CI completes successfully on `main`**; a red CI never
deploys. Uses a GitHub Actions **`environment: production`**.

1. Log in to GHCR and resolve the image tag from the triggering commit SHA.
2. `docker compose pull` — pull the new image from the registry.
3. `docker compose up -d --no-build` — deploy/update the running service.
4. Poll `docker inspect` until the container healthcheck reports `healthy` (fails after 120 s).
5. Run the smoke test.
6. Run the post-deployment monitoring batch (M5).
7. Upload service logs, metrics and the monitoring report as build artifacts.
8. On failure: dump logs and `docker compose down` (rollback).

### 3. Smoke Tests / Health Check — `scripts/smoke_test.py`

Four assertions, each exiting non-zero on failure, which fails the CD job:

1. `/health` returns 200 **and** `model_loaded: true` (with retry until timeout);
2. `/predict` returns 200 with a valid label, a confidence in [0, 1] and probabilities summing to 1;
3. a non-image upload is rejected with **400**, not 500 — the service must not crash on bad input;
4. `/metrics` exposes the request counters and the latency summary.

---

## M5 — Monitoring, Logs & Final Submission (10 marks)

### 1. Basic Monitoring & Logging

**Request/response logging** — an ASGI middleware in `app/main.py` emits one structured JSON
line per request and one per response:

```json
{"event": "request",  "request_id": "…", "method": "POST", "endpoint": "/predict"}
{"event": "response", "request_id": "…", "endpoint": "/predict", "status": 200, "latency_ms": 27.4}
{"event": "prediction", "content_type": "image/jpeg", "bytes": 13743,
 "content_sha256_12": "1c7a0e573db0", "label": "cat", "confidence": 0.9648, "inference_ms": 25}
```

**Sensitive data is excluded by construction.** Raw image bytes are never logged; only the
payload size and a 12-character SHA-256 prefix are recorded, which is enough to correlate a
prediction with a request without storing the image itself. Every response also carries
`X-Request-ID` and `X-Process-Time-ms` headers for tracing.

**Metrics** — `src/monitoring.py` maintains thread-safe in-app counters, exposed two ways:

| Metric | Meaning |
| --- | --- |
| `app_requests_total{endpoint}` | request count per endpoint |
| `app_responses_total{endpoint,status}` | response count per endpoint and status code |
| `app_predictions_total{label}` | predictions per predicted class (class-balance drift signal) |
| `app_errors_total{type}` | errors by type |
| `app_request_latency_ms{quantile}` | p50 / p95 / p99 latency, plus count and average |
| `app_uptime_seconds` | seconds since start |

`GET /metrics` serves Prometheus exposition format — scrapeable with no code change.
`GET /metrics/json` serves the same numbers as JSON, which is easier to read on camera.

### 2. Model Performance Tracking (Post-Deployment)

`scripts/monitor_batch.py` sends a small batch of **labelled** requests to the deployed
service and compares predictions with the true labels.

- Sample source: `data/monitoring_samples/{cat,dog}/` — a ~20-image labelled holdout drawn
  from the test split by `scripts/make_monitoring_set.py` and committed to the repository, so
  the check runs on a clean CI runner without the full dataset. Falls back to
  `data/processed/test/` when available.
- Computes: live accuracy, per-class recall, a confusion matrix, request failure count, and
  latency avg/p50/p95/max.
- Fetches the accuracy recorded at **training** time from `/model-info` and reports
  `accuracy_delta_vs_training`, raising `drift_flag` when live accuracy falls more than 10
  percentage points below it.
- Writes `reports/post_deploy_monitoring.json` and is uploaded as a CD build artifact.
- `--min-accuracy` turns the check into a hard gate when you want the pipeline to fail on
  degradation.

---

## Deliverables checklist

| Deliverable | Where |
| --- | --- |
| Source code | `src/`, `app/`, `tests/`, `scripts/` |
| DVC configuration | `dvc.yaml`, `dvc.lock`, `data/raw.dvc`, `.dvc/config` |
| CI/CD configuration | `.github/workflows/ci.yml`, `.github/workflows/cd.yml` |
| Docker configuration | `Dockerfile`, `.dockerignore`, `docker-compose.yml` |
| Deployment manifests | `docker-compose.yml` |
| Trained model artifact | `models/model.pt` |
| Training artifacts | `reports/confusion_matrix.png`, `reports/loss_curves.png`, `reports/classification_report.txt`, `reports/metrics.json` |
| Monitoring evidence | `reports/post_deploy_monitoring.json` |
| Submission zip | produced by `scripts/make_submission.ps1` |
| Screen recording (< 5 min) | shot list in `README.md` |

---

## Honest scope notes

- The trained artifacts shipped in this repository were produced on a **synthetic stand-in
  dataset** used to validate the pipeline end to end. Replace `data/raw` with the Kaggle
  dataset and re-run `python -m src.train` before submitting; accuracy figures in
  `reports/metrics.json` will then reflect real performance.
- The `docker build` step is the only stage that was not executed before this repository was
  handed over. Run it first (quickstart step 4).
- The CD job deploys to an ephemeral Compose environment on the GitHub-hosted runner and tears
  it down after the smoke test. That is a genuine automated deployment with a passing smoke
  gate; it is not a long-lived production host, which the assignment does not require.
