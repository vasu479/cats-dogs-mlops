<#
.SYNOPSIS
    Download the Kaggle Cats-vs-Dogs dataset and lay it out as data\raw\{cat,dog}.

.DESCRIPTION
    Requires the Kaggle API token. Get it from
        https://www.kaggle.com/settings  ->  API  ->  "Create New Token"
    and save the downloaded kaggle.json to
        $env:USERPROFILE\.kaggle\kaggle.json

    Default dataset: shaunthesheep/microsoft-catsvsdogs-dataset
      (the Microsoft Cats-vs-Dogs set - 25,000 images, no competition rules to
       accept, which makes it the least friction-prone option)

    Use -Dataset to point at a different Kaggle dataset slug.

.EXAMPLE
    .\scripts\download_data.ps1
    .\scripts\download_data.ps1 -Dataset "salader/dogs-vs-cats"
#>

[CmdletBinding()]
param(
    [string]$Dataset  = "shaunthesheep/microsoft-catsvsdogs-dataset",
    [string]$WorkDir  = "data\kaggle_download",
    [string]$RawDir   = "data\raw"
)

$ErrorActionPreference = "Stop"

function Write-Step($n, $text) { Write-Host "`n[$n] $text" -ForegroundColor Cyan }

# --- 1. Kaggle CLI ---------------------------------------------------------
Write-Step 1 "Checking the Kaggle CLI"
$kaggle = Get-Command kaggle -ErrorAction SilentlyContinue
if (-not $kaggle) {
    Write-Host "     kaggle CLI not found - installing into the active environment"
    python -m pip install --quiet kaggle
    $kaggle = Get-Command kaggle -ErrorAction SilentlyContinue
    if (-not $kaggle) { throw "Could not install the kaggle CLI. Run: pip install kaggle" }
}
Write-Host "     $($kaggle.Source)"

# --- 2. Credentials --------------------------------------------------------
Write-Step 2 "Checking Kaggle credentials"
$tokenPath = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if (-not (Test-Path $tokenPath)) {
    Write-Host @"
     kaggle.json not found at:
       $tokenPath

     1. Sign in at https://www.kaggle.com/settings
     2. API section -> 'Create New Token' (downloads kaggle.json)
     3. Move it to the path above, then re-run this script.

     No Kaggle account handy? Validate the pipeline with a synthetic set instead:
       python scripts\make_synthetic_data.py --per-class 500
"@ -ForegroundColor Yellow
    exit 1
}
Write-Host "     Found $tokenPath" -ForegroundColor Green

# --- 3. Download -----------------------------------------------------------
Write-Step 3 "Downloading '$Dataset' (this is a large download)"
if (-not (Test-Path $WorkDir)) { New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null }

kaggle datasets download -d $Dataset -p $WorkDir --unzip --force
if ($LASTEXITCODE -ne 0) { throw "Kaggle download failed. Check the dataset slug and your token." }
Write-Host "     Download and extraction complete" -ForegroundColor Green

# --- 4. Normalise the folder layout ---------------------------------------
Write-Step 4 "Laying images out as $RawDir\cat and $RawDir\dog"

$catPatterns = @("Cat", "cat", "cats", "Cats")
$dogPatterns = @("Dog", "dog", "dogs", "Dogs")

function Find-ClassFolders([string[]]$patterns) {
    Get-ChildItem -Path $WorkDir -Recurse -Directory |
        Where-Object { $patterns -contains $_.Name } |
        Sort-Object { $_.FullName.Length }
}

$catFolders = Find-ClassFolders $catPatterns
$dogFolders = Find-ClassFolders $dogPatterns

if (-not $catFolders -or -not $dogFolders) {
    Write-Host "     Could not auto-detect cat/dog folders. Contents of $WorkDir :" -ForegroundColor Yellow
    Get-ChildItem -Path $WorkDir -Recurse -Directory | Select-Object -First 25 FullName
    throw "Move the cat images into $RawDir\cat and the dog images into $RawDir\dog manually."
}

foreach ($pair in @(@{Src = $catFolders; Dest = "$RawDir\cat"}, @{Src = $dogFolders; Dest = "$RawDir\dog"})) {
    if (Test-Path $pair.Dest) { Remove-Item $pair.Dest -Recurse -Force }
    New-Item -ItemType Directory -Path $pair.Dest -Force | Out-Null

    $count = 0
    foreach ($folder in $pair.Src) {
        Get-ChildItem -Path $folder.FullName -File |
            Where-Object { $_.Extension -match '^\.(jpg|jpeg|png)$' } |
            ForEach-Object {
                Copy-Item $_.FullName (Join-Path $pair.Dest $_.Name) -Force
                $count++
            }
    }
    Write-Host "     $($pair.Dest): $count images"
}

# --- 5. Summary ------------------------------------------------------------
Write-Step 5 "Summary"
$catCount = (Get-ChildItem "$RawDir\cat" -File -ErrorAction SilentlyContinue).Count
$dogCount = (Get-ChildItem "$RawDir\dog" -File -ErrorAction SilentlyContinue).Count
Write-Host "     cat : $catCount images"
Write-Host "     dog : $dogCount images"

if ($catCount -lt 100 -or $dogCount -lt 100) {
    Write-Host "     WARNING: fewer than 100 images per class - check the extraction." -ForegroundColor Yellow
}

Write-Host "`n     Next:" -ForegroundColor Green
Write-Host "       dvc add data\raw"
Write-Host "       python -m src.data_prep"
Write-Host "       python -m src.train`n"
