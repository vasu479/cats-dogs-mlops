<#
.SYNOPSIS
    Build the submission zip: source, configuration, manifests and trained artifacts.

.DESCRIPTION
    Bundles everything the assignment asks for and deliberately leaves out the
    dataset (DVC-tracked, hundreds of MB) and the virtualenv.

    Include the MLflow run directory with -IncludeMlruns if you want the graders
    to be able to open the experiment locally; it can be tens of MB.

.EXAMPLE
    .\scripts\make_submission.ps1
    .\scripts\make_submission.ps1 -IncludeMlruns
#>

[CmdletBinding()]
param(
    [string]$Name = "2024AC05343_Sreenivasulu_Remuri_MLOps_Assignment2",
    [switch]$IncludeMlruns
)

$ErrorActionPreference = "Stop"

$staging = Join-Path $env:TEMP "cats-dogs-submission"
$zipPath = Join-Path (Get-Location) "$Name.zip"

Write-Host "`n[1] Pre-flight checks" -ForegroundColor Cyan

$required = @(
    "models\model.pt",
    "reports\confusion_matrix.png",
    "reports\loss_curves.png",
    "reports\metrics.json",
    "Dockerfile",
    "docker-compose.yml",
    "dvc.yaml",
    "requirements.txt",
    ".github\workflows\ci.yml",
    ".github\workflows\cd.yml"
)
$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing) {
    Write-Host "    Missing required deliverables:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "      $_" -ForegroundColor Red }
    Write-Host "    Run the pipeline (src.data_prep, src.train) before packaging." -ForegroundColor Red
    exit 1
}
Write-Host "    All required deliverables present" -ForegroundColor Green

if (-not (Test-Path "data\raw.dvc")) {
    Write-Host "    WARNING: data\raw.dvc missing - run 'dvc add data\raw' so DVC versioning is evidenced." -ForegroundColor Yellow
}

Write-Host "`n[2] Staging files" -ForegroundColor Cyan
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Path $staging | Out-Null

$include = @(
    "src", "app", "tests", "scripts", ".github",
    "models", "reports", "data",
    "Dockerfile", ".dockerignore", "docker-compose.yml",
    "dvc.yaml", "dvc.lock", "params.yaml",
    "requirements.txt", "requirements-dev.txt",
    "pytest.ini", "conftest.py",
    "README.md", "REPORT.md", "VALIDATION_RUNBOOK.md",
    ".gitignore", ".gitattributes", ".dvc"
)

foreach ($item in $include) {
    if (Test-Path $item) {
        Copy-Item $item -Destination $staging -Recurse -Force
        Write-Host "    + $item"
    }
}

if ($IncludeMlruns -and (Test-Path "mlruns")) {
    Copy-Item "mlruns" -Destination $staging -Recurse -Force
    Write-Host "    + mlruns (MLflow experiment history)"
}

Write-Host "`n[3] Removing bulk and caches from the bundle" -ForegroundColor Cyan
# The dataset is DVC-tracked; only the .dvc pointer and the tiny monitoring
# holdout belong in the zip.
foreach ($drop in @("data\raw", "data\processed", "data\kaggle_download")) {
    $path = Join-Path $staging $drop
    if (Test-Path $path) { Remove-Item $path -Recurse -Force; Write-Host "    - $drop (DVC-tracked)" }
}
Get-ChildItem $staging -Recurse -Directory -Include "__pycache__", ".pytest_cache", ".venv" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "    - __pycache__ / .pytest_cache / .venv"

Write-Host "`n[4] Creating archive" -ForegroundColor Cyan
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zipPath -CompressionLevel Optimal
Remove-Item $staging -Recurse -Force

$sizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
Write-Host "`n================ SUBMISSION READY ================" -ForegroundColor Green
Write-Host "  $zipPath  ($sizeMB MB)" -ForegroundColor Green
Write-Host @"

Still to do before you submit:
  1. Record the screen capture (< 5 minutes) - shot list is in README.md
  2. Confirm the GitHub repo is accessible to the graders
  3. Confirm the GHCR package page shows the published image tags

"@ -ForegroundColor Green
