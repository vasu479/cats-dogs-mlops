# Cats vs Dogs — End-to-End MLOps Pipeline

**MLOps (S1-25_AIMLCZG523) · Assignment 2**
**Student:** Sreenivasulu Remuri · **BITS ID:** 2024AC05343

Binary image classification (cats vs dogs) for a pet adoption platform, delivered as a
complete MLOps pipeline: data versioning → experiment tracking → containerised inference →
CI → CD → monitoring.

---



## Architecture

```
                       ┌──────────────────────────────────────────┐
   Kaggle dataset ───► │  M1  data_prep.py → 224×224 RGB          │
   data/raw/{cat,dog}  │      80/10/10 split + augmentation       │
        │              │      train.py → SimpleCNN → model.pt     │
     [ DVC ]           │      MLflow: params, metrics, artifacts  │
        │              └───────────────────┬──────────────────────┘
        ▼                                  │
   dvc.yaml pipeline                       ▼
                       ┌──────────────────────────────────────────┐
                       │  M2  FastAPI  /health  /predict          │
                       │      requirements.txt (pinned)           │
                       │      Dockerfile (multi-stage, non-root)  │
                       └───────────────────┬──────────────────────┘
                                           │  git push
                                           ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  M3  GitHub Actions — CI                                      │
   │      checkout → install → pytest (44 tests) → docker build     │
   │      → push  ghcr.io/vasu479/cats-dogs-api:{sha,latest}        │
   └───────────────────────────┬───────────────────────────────────┘
                               │  workflow_run: CI succeeded on main
                               ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  M4  GitHub Actions — CD  (environment: production)           │
   │      docker compose pull → up -d → wait healthy                │
   │      → smoke_test.py   ← pipeline FAILS here if it fails       │
   └───────────────────────────┬───────────────────────────────────┘
                               ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  M5  Monitoring — structured JSON request/response logs,       │
   │      /metrics (Prometheus) + /metrics/json,                    │
   │      monitor_batch.py → live accuracy vs training accuracy     │
   └───────────────────────────────────────────────────────────────┘
```

---

## Repository layout

```
cats-dogs-mlops/
├── src/
│   ├── config.py              typed access to params.yaml
│   ├── data_prep.py           M1  224×224 RGB, 80/10/10 split          [unit tested]
│   ├── datasets.py            M1  augmentation + DataLoaders
│   ├── model.py               M1  SimpleCNN baseline                    [unit tested]
│   ├── train.py               M1  training loop + MLflow tracking
│   ├── inference.py           M2  model loading and prediction          [unit tested]
│   └── monitoring.py          M5  request/latency counters
├── app/main.py                M2  FastAPI service + M5 logging middleware
├── tests/
│   ├── test_data_prep.py      M3  pre-processing unit tests
│   ├── test_model_utils.py    M3  inference-utility unit tests
│   └── test_api.py            M3  API integration tests
├── scripts/
│   ├── setup_windows.ps1      one-time environment setup
│   ├── download_data.ps1      Kaggle download + folder normalisation
│   ├── make_synthetic_data.py synthetic stand-in dataset
│   ├── make_monitoring_set.py builds the committed labelled holdout
│   ├── docker_cleanup.ps1     scoped (or full) Docker reset
│   ├── run_local_demo.ps1     the screen-recording script
│   ├── smoke_test.py          M4  post-deploy smoke test
│   └── push_to_github.ps1     pre-push checks + push
├── .github/workflows/
│   ├── ci.yml                 M3  test → build → push to GHCR
│   └── cd.yml                 M4  pull → deploy → smoke test → monitor
├── Dockerfile                 M2  multi-stage, non-root, healthcheck
├── docker-compose.yml         M4  deployment target
├── dvc.yaml / params.yaml     M1  reproducible pipeline + hyper-parameters
├── requirements.txt           M2  pinned runtime dependencies
├── requirements-dev.txt       pinned training/test dependencies
└── .gitattributes             Windows CRLF safety (see Troubleshooting)
```

---

### 0. Clean Docker first

```powershell

.\scripts\docker_cleanup.ps1

# Preview without deleting anything:
.\scripts\docker_cleanup.ps1 -WhatIf


.\scripts\docker_cleanup.ps1 -Full
```

### 1. Set up the environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # allow local scripts this session
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```

### 2. Get the dataset

```powershell
# Real data (needs %USERPROFILE%\.kaggle\kaggle.json):
.\scripts\download_data.ps1


python scripts\make_synthetic_data.py --per-class 500
```

### 3. M1 — version data, preprocess, train, track

```powershell
dvc add data\raw                 # version the raw dataset
git add data\raw.dvc .gitignore

python -m src.data_prep          # → 224×224 RGB, 80/10/10 split
python -m src.train              # → models\model.pt + MLflow run + reports\*.png

python scripts\make_monitoring_set.py --per-class 10   # labelled holdout for M5

mlflow ui --backend-store-uri file:./mlruns            # → http://127.0.0.1:5000
```

Reproduce the whole pipeline through DVC instead:

```powershell
dvc repro
dvc dag
dvc metrics show
dvc push                          # to the local remote created by setup_windows.ps1
```

### 4. M2 — containerise and verify locally

```powershell
docker compose build
docker compose up -d
docker compose ps

curl.exe http://localhost:8000/health
curl.exe -X POST http://localhost:8000/predict -F "file=@data\monitoring_samples\cat\<some>.jpg"
```

Interactive API docs: <http://localhost:8000/docs>

### 5. M3 + M4 — push, and let CI/CD run

```powershell
pytest                                                     # 44 tests, all green locally first

.\scripts\push_to_github.ps1 -RepoUrl "https://github.com/vasu479/cats-dogs-mlops.git"
```

**One-time GitHub setup** (before the first push):

1. `Settings → Actions → General → Workflow permissions` → **Read and write permissions** → Save.
   Without this, the CI job cannot push to GHCR.
2. After the first successful CI run, open
   `https://github.com/vasu479/cats-dogs-mlops/pkgs/container/cats-dogs-api`
   → *Package settings* → set visibility (public makes the demo pull simplest).

CI runs on every push. When CI succeeds on `main`, CD triggers automatically.

### 6. Deploy the CI-built image locally (optional, good for the recording)

```powershell
docker login ghcr.io -u vasu479          # use a PAT with read:packages
.\scripts\run_local_demo.ps1 -Image "ghcr.io/vasu479/cats-dogs-api:latest"
```

### 7. M5 — monitoring

```powershell
curl.exe http://localhost:8000/metrics            # Prometheus format
curl.exe http://localhost:8000/metrics/json       # same counters, JSON

python scripts\monitor_batch.py --base-url http://localhost:8000 --samples 20

docker compose logs cats-dogs-api                 # structured JSON request/response logs
```

---

## API reference

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | **M2 endpoint 1.** Liveness + `model_loaded` readiness flag. Always 200. |
| `GET` | `/ready` | Strict readiness: 503 until the checkpoint is loaded. |
| `POST` | `/predict` | **M2 endpoint 2.** Multipart image → label + class probabilities. |
| `GET` | `/metrics` | **M5.** Prometheus exposition text. |
| `GET` | `/metrics/json` | **M5.** Same counters as JSON. |
| `GET` | `/model-info` | Checkpoint metadata and training metrics. |
| `GET` | `/docs` | Swagger UI. |

Example `/predict` response:

```json
{
  "label": "cat",
  "label_index": 0,
  "confidence": 0.964779,
  "probabilities": { "cat": 0.964779, "dog": 0.035221 },
  "inference_time_ms": 25
}
```

---

## Design decisions worth defending

**Why `models/model.pt` is in Git and not DVC.**
The baseline checkpoint is ~1 MB. The Docker image bakes it in, so `docker build` on a clean
GitHub runner must be able to see it. Putting it in DVC would force every CI run to
authenticate against a DVC remote for a one-megabyte file. DVC tracks what it is good at —
the multi-hundred-megabyte `data/raw` and `data/processed` trees. `dvc.yaml` still records the
checkpoint's hash via `cache: false`, so a change in the model is still visible in `dvc.lock`.

**Why Docker Compose rather than Kubernetes.**
M4 permits Compose, Kubernetes or a VM. Compose has the fewest moving parts on Windows, and
the *same* `docker-compose.yml` is used by the CD job on the runner and by the laptop demo —
so what CI proves is exactly what gets demonstrated.

**Why GHCR rather than Docker Hub.**
`GITHUB_TOKEN` is minted per-run and scoped to the repository. Nothing has to be created,
rotated, or kept out of frame during a screen recording.

**Why the runtime image excludes MLflow, scikit-learn and matplotlib.**
They are training-time dependencies. Excluding them roughly halves the image and shrinks the
attack surface of the deployed service.

**Why augmentation is train-split only.**
Validation and test use the deterministic resize path, so metrics remain comparable between
runs and the confusion matrix reflects real performance rather than augmentation noise.

---

## Troubleshooting (Windows-specific)

| Symptom | Cause | Fix |
| --- | --- | --- |
| `exec /usr/bin/sh: no such file or directory` in the container | Git checked files out with CRLF | `.gitattributes` already forces LF. Run `git config core.autocrlf false`, then `git rm --cached -r . && git reset --hard`. |
| `docker compose build` pulls a ~2.5 GB torch wheel | The CPU index was not used | The Dockerfile passes `--extra-index-url https://download.pytorch.org/whl/cpu`. Do not remove it. |
| `docker compose build` hangs silently for minutes inside `pip install` | Your network (VPN, corporate proxy) blocks `download.pytorch.org`. pip retries quietly instead of failing fast — verified behaviour, not a guess. | `curl.exe -I https://download.pytorch.org/whl/cpu/` to confirm, then build off the VPN. |
| `.\scripts\...ps1 cannot be loaded` | PowerShell execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| CI fails at *Verify the trained model artifact* | `models/model.pt` was never committed | `python -m src.train`, then commit `models/model.pt`. |
| CI cannot push to `ghcr.io` (403) | Workflow permissions are read-only | `Settings → Actions → General → Workflow permissions → Read and write`. |
| Training hangs at 0% on Windows | `num_workers > 0` with the spawn start method | `params.yaml` sets `train.num_workers: 0`. Keep it there. |
| Port 8000 already in use | Another container is bound | `docker compose down`, or `$env:HOST_PORT="8001"` before `docker compose up`. |
| `/predict` returns 503 | The container has no model | Confirm `models/model.pt` existed at build time; check `docker compose logs`. |

---


---

## Verification status

Executed and passing on Linux/CPU before delivery:

- `python -m src.data_prep` → 320/40/40 split, 0 corrupt
- `python -m src.train` → MLflow run created, `model.pt` + confusion matrix + loss curves written
- `pytest` → **44 passed**
- `scripts/smoke_test.py` → all 4 checks passed against the live service
- `scripts/monitor_batch.py` → 20/20 requests, p50 latency 29 ms
- All YAML (`ci.yml`, `cd.yml`, `docker-compose.yml`, `dvc.yaml`, `params.yaml`) parses

Not executed before delivery: `docker build` (no Docker daemon was available in the authoring
environment). The Dockerfile was reviewed statically and its runtime import graph verified
against `requirements.txt`. 
no prior execution behind it.
