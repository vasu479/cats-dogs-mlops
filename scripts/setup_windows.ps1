<#
.SYNOPSIS
    One-time environment setup on Windows: virtualenv, dependencies, Git, DVC.

.DESCRIPTION
    Idempotent - safe to re-run. Creates .venv, installs pinned dependencies,
    initialises Git and DVC, and configures a local DVC remote.

.EXAMPLE
    .\scripts\setup_windows.ps1
    .\scripts\setup_windows.ps1 -SkipDvc
#>

[CmdletBinding()]
param(
    [string]$PythonExe = "python",
    [switch]$SkipDvc
)

$ErrorActionPreference = "Stop"

function Write-Step($n, $text) {
    Write-Host "`n[$n] $text" -ForegroundColor Cyan
}

# --- 1. Python -------------------------------------------------------------
Write-Step 1 "Checking Python"
$version = & $PythonExe --version 2>&1
Write-Host "     $version"
if ($version -notmatch "3\.(11|12|13)") {
    Write-Host "     WARNING: Python 3.11-3.13 is recommended. Pinned wheels may not exist for $version." -ForegroundColor Yellow
}

# --- 2. Virtual environment ------------------------------------------------
Write-Step 2 "Creating virtual environment (.venv)"
if (-not (Test-Path ".venv")) {
    & $PythonExe -m venv .venv
    Write-Host "     Created .venv"
} else {
    Write-Host "     .venv already exists - reusing"
}

$venvPython = Join-Path (Resolve-Path ".venv") "Scripts\python.exe"
if (-not (Test-Path $venvPython)) { throw "Could not find $venvPython" }

# --- 3. Dependencies -------------------------------------------------------
Write-Step 3 "Installing pinned dependencies (this takes a few minutes)"
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install --extra-index-url https://download.pytorch.org/whl/cpu `
    -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
Write-Host "     Dependencies installed" -ForegroundColor Green

& $venvPython -c "import torch, torchvision, fastapi, mlflow; print('     torch', torch.__version__, '| torchvision', torchvision.__version__, '| mlflow', mlflow.__version__)"

# --- 4. Git ----------------------------------------------------------------
Write-Step 4 "Initialising Git"
if (-not (Test-Path ".git")) {
    git init -b main
    Write-Host "     Git repository initialised on branch 'main'"
} else {
    Write-Host "     Git repository already initialised"
}

# Line endings: .gitattributes forces LF for everything the container runs.
git config core.autocrlf false
Write-Host "     core.autocrlf = false (LF enforced via .gitattributes)"

# --- 5. DVC ----------------------------------------------------------------
if (-not $SkipDvc) {
    Write-Step 5 "Initialising DVC"
    if (-not (Test-Path ".dvc")) {
        & $venvPython -m dvc init
        Write-Host "     DVC initialised"
    } else {
        Write-Host "     DVC already initialised"
    }

    $remotePath = Join-Path (Split-Path (Get-Location) -Parent) "dvc-storage"
    if (-not (Test-Path $remotePath)) { New-Item -ItemType Directory -Path $remotePath | Out-Null }
    & $venvPython -m dvc remote add -d -f localremote $remotePath
    Write-Host "     DVC remote 'localremote' -> $remotePath"
} else {
    Write-Step 5 "Skipping DVC (-SkipDvc)"
}

# --- 6. Folders ------------------------------------------------------------
Write-Step 6 "Creating working folders"
foreach ($dir in @("data\raw\cat", "data\raw\dog", "models", "reports", "logs")) {
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
}
Write-Host "     data\raw\{cat,dog}, models, reports, logs"

Write-Host "`n================ SETUP COMPLETE ================" -ForegroundColor Green
Write-Host @"

Activate the environment in every new terminal:

    .\.venv\Scripts\Activate.ps1

Then get the dataset:

    .\scripts\download_data.ps1                 # real Kaggle data
    python scripts\make_synthetic_data.py       # or a synthetic stand-in

"@ -ForegroundColor Green
