# Validation Runbook — run these in order

**Student:** Sreenivasulu Remuri · **BITS ID:** 2024AC05343
**Environment:** Windows + PowerShell + Docker Desktop

Nine gates. Each has a **pass condition**. If a gate fails, stop and fix it — do not carry a
broken state into the next gate. Total time: roughly 45–70 minutes, most of it downloads.

Open PowerShell (**not** cmd), `cd` into the unzipped `cats-dogs-mlops` folder, and start.

---

## Gate 0 — Docker is clean  ·  ~1 min

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

docker version                                  # daemon must respond
.\scripts\docker_cleanup.ps1 -WhatIf            # preview: deletes nothing
.\scripts\docker_cleanup.ps1                    # scoped cleanup
```

**Pass:** the final table lists your other images (e.g. `heart-disease-api`) and **no**
`cats-dogs*` image, container, volume or network.

> Only if you truly want a machine-wide wipe: `.\scripts\docker_cleanup.ps1 -Full`.
> It asks you to type `WIPE EVERYTHING` and it will delete your Assignment 1 images.

---

## Gate 1 — Environment  ·  ~5–10 min

```powershell
python --version                                # expect 3.11, 3.12 or 3.13

.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1

python -c "import torch, torchvision, fastapi, mlflow, dvc; print('imports OK')"
dvc --version
```

**Pass:** `imports OK` prints, and `dvc --version` prints a version.

> If `dvc --version` throws `cannot import name '_DIR_MARK'`, the pathspec pin did not take.
> Fix: `pip install pathspec==0.12.1`

---

## Gate 2 — Dataset  ·  ~10–20 min (download-bound)

```powershell
# Real Kaggle data. Needs %USERPROFILE%\.kaggle\kaggle.json first.
.\scripts\download_data.ps1

# Confirm the layout
(Get-ChildItem data\raw\cat -File).Count
(Get-ChildItem data\raw\dog -File).Count
```

**Pass:** both counts are in the thousands.

> No Kaggle token yet and you want to test the plumbing now:
> `python scripts\make_synthetic_data.py --per-class 500`
> Then come back and redo this gate with real data before submitting.

---

## Gate 3 — M1: version, preprocess, train, track  ·  ~15–40 min

```powershell
dvc add data\raw
git add data\raw.dvc data\.gitignore

python -m src.data_prep
Get-Content data\processed\summary.json

python -m src.train

python scripts\make_monitoring_set.py --per-class 10
```

**Pass:**

- `summary.json` shows an ~80/10/10 split and `image_size: 224`
- `models\model.pt` exists
- `reports\confusion_matrix.png`, `reports\loss_curves.png`, `reports\metrics.json` exist

Then look at the experiment and take your M1 screenshot:

```powershell
mlflow ui --backend-store-uri file:./mlruns      # http://127.0.0.1:5000  (Ctrl+C to stop)
```

**Screenshot:** the run page showing parameters, the metric curves, and the Artifacts tab
with the confusion matrix and loss curves.

Optionally prove the DVC pipeline reproduces:

```powershell
dvc dag
dvc repro
dvc metrics show
```

> Training too slow? Lower `data.max_per_class` in `params.yaml` (2000 → 800) and re-run.
> `train.num_workers` must stay `0` on Windows.

---

## Gate 4 — M3: tests pass locally  ·  ~1 min

```powershell
pytest
```

**Pass:** `44 passed`. Never push a red suite — CI will just reject it more slowly.

---

## Gate 5 — M2: build and run the container  ⚠️ ~5–10 min

**This is the one step with no prior execution behind it. Everything else was verified.**

```powershell
docker compose build
docker compose up -d
docker compose ps
```

**Pass:** `docker compose ps` shows `cats-dogs-api` as `running (healthy)`. Wait up to 45s
for the health status to flip from `starting`.

```powershell
# Health check (M2 endpoint 1)
curl.exe http://localhost:8000/health
```

**Pass:** JSON with `"status":"healthy"` and `"model_loaded":true`.

```powershell
# Prediction (M2 endpoint 2) — pick any real image
$img = (Get-ChildItem data\processed\test\cat -File | Select-Object -First 1).FullName
curl.exe -X POST http://localhost:8000/predict -F "file=@$img"
```

**Pass:** JSON with `label`, `confidence`, and a `probabilities` object holding both classes.

Also open <http://localhost:8000/docs> — good footage for the recording.

**If this gate fails**, these are the likely causes in order:

| Error | Cause | Fix |
| --- | --- | --- |
| Build hangs for minutes with no output during `pip install` | Your network blocks or throttles `download.pytorch.org`; pip retries silently rather than failing | Check on a different network / disable VPN. Confirm with `curl.exe -I https://download.pytorch.org/whl/cpu/` |
| `exec /usr/bin/sh: no such file or directory` | Files checked out with CRLF | `git config core.autocrlf false`, then `git rm --cached -r .` and `git reset --hard` |
| `COPY models/: not found` | You skipped Gate 3 | Run `python -m src.train` |
| `/health` says `model_loaded:false` | `models/model.pt` was absent at build time | Rebuild after Gate 3: `docker compose build --no-cache` |
| Port 8000 in use | Something else is bound | `$env:HOST_PORT="8001"` then `docker compose up -d` |

---

## Gate 6 — M4 + M5 locally  ·  ~2 min

```powershell
python scripts\smoke_test.py --base-url http://localhost:8000
```

**Pass:** four `[ OK ]` lines and `=== ALL SMOKE TESTS PASSED ===`.

```powershell
python scripts\monitor_batch.py --base-url http://localhost:8000 --samples 20
```

**Pass:** `failed requests: 0`, and a live accuracy close to your training accuracy.

```powershell
curl.exe http://localhost:8000/metrics          # Prometheus format
curl.exe http://localhost:8000/metrics/json     # same counters as JSON
docker compose logs cats-dogs-api               # structured JSON request/response logs
```

**Pass:** `app_requests_total`, `app_predictions_total` and `app_request_latency_ms` appear,
and the logs show `{"event":"prediction", ...}` lines with **no raw image data** — only a size
and a hash prefix.

You can now run the whole demo in one command for the recording:

```powershell
.\scripts\run_local_demo.ps1
```

---

## Gate 7 — M3: push and watch CI  ·  ~5 min

**One-time GitHub setup, before the first push:**

1. Create an empty repo `cats-dogs-mlops` at <https://github.com/new> (no README, no .gitignore).
2. `Settings → Actions → General → Workflow permissions` → **Read and write permissions** → Save.
   Skip this and the CI job cannot push to GHCR.

```powershell
docker compose down                              # free the port before pushing

.\scripts\push_to_github.ps1 -RepoUrl "https://github.com/vasu479/cats-dogs-mlops.git"
```

The script re-runs the pre-flight checks, blocks any file over 50 MB, commits and pushes.

Then open `https://github.com/vasu479/cats-dogs-mlops/actions`.

**Pass:** the **CI** run is green — job `test` (44 tests) then `build-and-push`, ending with
a job summary naming the two published tags.

Confirm the image landed:
`https://github.com/vasu479/cats-dogs-mlops/pkgs/container/cats-dogs-api`

> `denied: permission_granted: false` at the GHCR push step means step 2 above was skipped.

---

## Gate 8 — M4: CD runs automatically  ·  ~5 min

Nothing to type. When CI succeeds on `main`, the **CD** workflow triggers itself.

Open the CD run and confirm each step:

1. Resolve image reference
2. Pull the new image from the registry
3. Deploy / update the running service
4. Wait for the container to report healthy
5. **Post-deploy smoke test** ← the gate
6. Post-deploy model performance check (M5)

**Pass:** CD is green, and the `deploy-evidence` artifact at the bottom of the run contains
`service.log`, `metrics.txt` and `post_deploy_monitoring.json`.

**Prove the gate actually gates** — this is the single best thing to show a grader:

```powershell
git checkout -b break-smoke
# In app\main.py, change the /health handler to return "status": "broken"
git add app\main.py
git commit -m "demo: prove the smoke test gates deployment"
git push -u origin break-smoke
# Open a PR to main, merge it, watch CD fail at the smoke test, then revert.
```

---

## Gate 9 — Package the submission  ·  ~2 min

```powershell
# Delete the bootstrap markers now that your artifacts are real
Remove-Item models\BOOTSTRAP_PLACEHOLDER.md, data\monitoring_samples\BOOTSTRAP_PLACEHOLDER.md

.\scripts\make_submission.ps1 -IncludeMlruns
```

**Pass:** `2024AC05343_Sreenivasulu_Remuri_MLOps_Assignment2.zip` is created and the pre-flight
check reports all required deliverables present.

Then record the screen capture — shot list is in `README.md`. Record the
**editor → commit → CI → CD → deployed prediction** segment without cuts; it is the most
persuasive 105 seconds in the video.

---

## Quick reference

```powershell
docker compose up -d                  # start
docker compose down                   # stop
docker compose logs -f cats-dogs-api  # follow logs
docker compose build --no-cache       # rebuild from scratch
.\scripts\docker_cleanup.ps1          # scoped reset
pytest                                # 44 tests
python -m src.train                   # retrain
mlflow ui --backend-store-uri file:./mlruns
```
