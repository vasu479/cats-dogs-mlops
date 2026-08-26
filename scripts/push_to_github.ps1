<#
.SYNOPSIS
    Commit the project and push it to GitHub, which triggers CI and then CD.

.DESCRIPTION
    Verifies the repository is in a submittable state (model artifact present,
    tests green, no dataset accidentally staged), then commits and pushes.

.EXAMPLE
    .\scripts\push_to_github.ps1 -RepoUrl "https://github.com/vasu479/cats-dogs-mlops.git"
    .\scripts\push_to_github.ps1 -Message "M3: add CI pipeline"
    .\scripts\push_to_github.ps1 -SkipTests
#>

[CmdletBinding()]
param(
    [string]$RepoUrl,
    [string]$Branch  = "main",
    [string]$Message = "MLOps Assignment 2: end-to-end cats-vs-dogs pipeline",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

function Banner($n, $text) {
    Write-Host "`n[$n] $text" -ForegroundColor Cyan
}

# --- 1. Repository ---------------------------------------------------------
Banner 1 "Git repository"
if (-not (Test-Path ".git")) {
    git init -b $Branch
    Write-Host "     Initialised on branch '$Branch'"
}
git config core.autocrlf false

# --- 2. Pre-push checks ----------------------------------------------------
Banner 2 "Pre-push checks"

if (-not (Test-Path "models\model.pt")) {
    Write-Host "     models\model.pt is missing. CI builds the image from it and WILL fail." -ForegroundColor Red
    Write-Host "     Run: python -m src.train" -ForegroundColor Red
    exit 1
}
$modelMB = [math]::Round((Get-Item "models\model.pt").Length / 1MB, 2)
Write-Host "     models\model.pt present ($modelMB MB)" -ForegroundColor Green
if ($modelMB -gt 90) {
    Write-Host "     WARNING: over GitHub's 100 MB per-file limit is near. Use Git LFS or DVC for this artifact." -ForegroundColor Yellow
}

if (-not (Test-Path "data\monitoring_samples\cat")) {
    Write-Host "     data\monitoring_samples missing - the CD monitoring step will fail." -ForegroundColor Yellow
    Write-Host "     Run: python scripts\make_monitoring_set.py --per-class 10" -ForegroundColor Yellow
}

if (-not $SkipTests) {
    Write-Host "     Running pytest..."
    pytest -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host "     Tests failed. Fix them before pushing - CI will reject this commit." -ForegroundColor Red
        exit 1
    }
    Write-Host "     Tests passed" -ForegroundColor Green
}

# --- 3. Guard against committing the dataset ------------------------------
Banner 3 "Checking nothing oversized is staged"
git add -A
$large = git diff --cached --name-only | ForEach-Object {
    if (Test-Path $_) {
        $item = Get-Item $_
        if ($item.Length -gt 50MB) { "$($_) ($([math]::Round($item.Length/1MB,1)) MB)" }
    }
}
if ($large) {
    Write-Host "     These staged files exceed 50 MB:" -ForegroundColor Red
    $large | ForEach-Object { Write-Host "       $_" -ForegroundColor Red }
    Write-Host "     Datasets belong in DVC, not Git. Unstage them and retry." -ForegroundColor Red
    git reset | Out-Null
    exit 1
}
Write-Host "     No oversized files staged" -ForegroundColor Green

$staged = git diff --cached --name-only
Write-Host "     $((($staged) | Measure-Object).Count) files staged"

# --- 4. Commit -------------------------------------------------------------
Banner 4 "Commit"
if ($staged) {
    git commit -m $Message
    Write-Host "     Committed: $Message" -ForegroundColor Green
} else {
    Write-Host "     Nothing to commit - working tree is clean"
}

# --- 5. Remote -------------------------------------------------------------
Banner 5 "Remote"
$existingRemotes = @(git remote)
$existing = if ($existingRemotes -contains "origin") { git remote get-url origin } else { $null }
if ($RepoUrl) {
    if ($existing) { git remote set-url origin $RepoUrl } else { git remote add origin $RepoUrl }
    Write-Host "     origin -> $RepoUrl"
} elseif ($existing) {
    Write-Host "     origin -> $existing"
} else {
    Write-Host "     No remote configured. Re-run with -RepoUrl https://github.com/<you>/cats-dogs-mlops.git" -ForegroundColor Red
    exit 1
}

# --- 6. Push ---------------------------------------------------------------
Banner 6 "Push to $Branch"
git branch -M $Branch
git push -u origin $Branch
if ($LASTEXITCODE -ne 0) { throw "git push failed." }

$remote = git remote get-url origin
$webUrl = $remote -replace '\.git$', ''

Write-Host "`n================ PUSHED ================" -ForegroundColor Green
Write-Host @"

CI is running now:  $webUrl/actions

Once CI is green, CD triggers automatically and deploys the image.

One-time GHCR setup (only needed the first time):
  1. $webUrl/settings/actions
     -> Workflow permissions -> "Read and write permissions" -> Save
  2. After the first successful push, make the package visible:
     $webUrl/pkgs/container/cats-dogs-api -> Package settings -> Change visibility

"@ -ForegroundColor Green
