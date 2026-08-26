<#
.SYNOPSIS
    End-to-end local demo: clean Docker, build, deploy, verify, monitor.

.DESCRIPTION
    This is the script to run while recording the 5-minute submission video.
    Each step prints a labelled banner so the recording is easy to follow.

.EXAMPLE
    .\scripts\run_local_demo.ps1
    .\scripts\run_local_demo.ps1 -SkipCleanup
    .\scripts\run_local_demo.ps1 -Image "ghcr.io/vasu479/cats-dogs-api:latest"   # deploy the CI image
#>

[CmdletBinding()]
param(
    [string]$Image,
    [int]$Port = 8000,
    [switch]$SkipCleanup
)

$ErrorActionPreference = "Stop"
$BaseUrl = "http://localhost:$Port"

function Banner($n, $text) {
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor Cyan
    Write-Host "  STEP $n :: $text" -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor Cyan
}

# --- 0 ---------------------------------------------------------------------
if (-not $SkipCleanup) {
    Banner 0 "Clean Docker state (scoped to this project)"
    & "$PSScriptRoot\docker_cleanup.ps1"
} else {
    Banner 0 "Skipping cleanup (-SkipCleanup)"
}

# --- 1 ---------------------------------------------------------------------
Banner 1 "Verify the trained model artifact exists"
if (-not (Test-Path "models\model.pt")) {
    Write-Host "  models\model.pt not found. Run 'python -m src.train' first." -ForegroundColor Red
    exit 1
}
$size = [math]::Round((Get-Item "models\model.pt").Length / 1MB, 2)
Write-Host "  models\model.pt  ($size MB)" -ForegroundColor Green

# --- 2 ---------------------------------------------------------------------
if ($Image) {
    Banner 2 "Pull the CI-built image from GHCR"
    $env:IMAGE = $Image
    docker compose pull
    if ($LASTEXITCODE -ne 0) { throw "docker compose pull failed." }
} else {
    Banner 2 "Build the container image locally"
    docker compose build --no-cache
    if ($LASTEXITCODE -ne 0) { throw "docker compose build failed." }
}
docker images --filter "reference=*cats-dogs*" --format "table {{.Repository}}`t{{.Tag}}`t{{.Size}}"

# --- 3 ---------------------------------------------------------------------
Banner 3 "Deploy with Docker Compose"
$env:HOST_PORT = "$Port"
docker compose up -d
if ($LASTEXITCODE -ne 0) { throw "docker compose up failed." }
docker compose ps

# --- 4 ---------------------------------------------------------------------
Banner 4 "Wait for the container health check to pass"
$healthy = $false
for ($i = 1; $i -le 45; $i++) {
    $status = docker inspect --format '{{.State.Health.Status}}' cats-dogs-api 2>$null
    Write-Host "  attempt $i : $status"
    if ($status -eq "healthy") { $healthy = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $healthy) {
    Write-Host "  Container never became healthy. Logs:" -ForegroundColor Red
    docker compose logs --tail 60
    exit 1
}
Write-Host "  Container is healthy" -ForegroundColor Green

# --- 5 ---------------------------------------------------------------------
Banner 5 "GET /health"
Invoke-RestMethod -Uri "$BaseUrl/health" | ConvertTo-Json -Depth 5

# --- 6 ---------------------------------------------------------------------
Banner 6 "POST /predict"
$sample = Get-ChildItem "data\monitoring_samples" -Recurse -File -Filter *.jpg -ErrorAction SilentlyContinue |
          Select-Object -First 1
if (-not $sample) {
    $sample = Get-ChildItem "data\processed\test" -Recurse -File -Filter *.jpg -ErrorAction SilentlyContinue |
              Select-Object -First 1
}
if ($sample) {
    Write-Host "  Sending: $($sample.FullName)"
    Write-Host "`n  --- curl equivalent ---" -ForegroundColor DarkGray
    Write-Host "  curl.exe -X POST $BaseUrl/predict -F `"file=@$($sample.FullName)`"" -ForegroundColor DarkGray
    Write-Host ""
    curl.exe -s -X POST "$BaseUrl/predict" -F "file=@$($sample.FullName)"
    Write-Host ""
} else {
    Write-Host "  No sample image found - the smoke test below sends a generated one." -ForegroundColor Yellow
}

# --- 7 ---------------------------------------------------------------------
Banner 7 "Post-deploy smoke test (M4 - fails the pipeline if this fails)"
python scripts\smoke_test.py --base-url $BaseUrl --timeout 60
if ($LASTEXITCODE -ne 0) { Write-Host "  SMOKE TEST FAILED" -ForegroundColor Red; exit 1 }

# --- 8 ---------------------------------------------------------------------
Banner 8 "Post-deployment model performance (M5)"
python scripts\monitor_batch.py --base-url $BaseUrl --samples 20
if ($LASTEXITCODE -ne 0) { Write-Host "  Monitoring batch reported failures" -ForegroundColor Yellow }

# --- 9 ---------------------------------------------------------------------
Banner 9 "Live metrics (M5)"
Write-Host "`n  --- Prometheus format ---" -ForegroundColor DarkGray
(Invoke-WebRequest -Uri "$BaseUrl/metrics").Content
Write-Host "`n  --- JSON ---" -ForegroundColor DarkGray
Invoke-RestMethod -Uri "$BaseUrl/metrics/json" | ConvertTo-Json -Depth 5

# --- 10 --------------------------------------------------------------------
Banner 10 "Structured request/response logs (M5)"
docker compose logs --tail 25 cats-dogs-api

Write-Host "`n" ("=" * 72) -ForegroundColor Green
Write-Host "  DEMO COMPLETE - service live at $BaseUrl (docs at $BaseUrl/docs)" -ForegroundColor Green
Write-Host "  Stop it with:  docker compose down" -ForegroundColor Green
Write-Host ("=" * 72) -ForegroundColor Green
Write-Host ""
