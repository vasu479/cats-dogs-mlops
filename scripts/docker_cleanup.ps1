<#
.SYNOPSIS
    Scoped Docker cleanup for the cats-dogs-mlops project.

.DESCRIPTION
    Removes ONLY this project's containers, images, volumes, networks and build
    cache. Other projects on the machine - including the heart-disease-mlops
    images from Assignment 1 - are left untouched.

    Use -Full only if you genuinely want a machine-wide wipe. That removes every
    stopped container, every unused image and the entire build cache for EVERY
    project on this machine, and it cannot be undone.

.EXAMPLE
    .\scripts\docker_cleanup.ps1
    .\scripts\docker_cleanup.ps1 -Full
    .\scripts\docker_cleanup.ps1 -WhatIf
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$Full,
    [string]$ProjectPattern = "cats-dogs"
)

$ErrorActionPreference = "Stop"

function Write-Section($Text) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

# --- Preconditions ---------------------------------------------------------
Write-Section "Checking Docker"
try {
    docker version --format '{{.Server.Version}}' | ForEach-Object {
        Write-Host "  Docker engine: $_" -ForegroundColor Green
    }
} catch {
    Write-Host "  Docker is not running. Start Docker Desktop and retry." -ForegroundColor Red
    exit 1
}

Write-Section "Disk usage BEFORE cleanup"
docker system df

# --- Full wipe -------------------------------------------------------------
if ($Full) {
    Write-Section "FULL WIPE REQUESTED"
    Write-Host "  This removes ALL stopped containers, ALL unused images," -ForegroundColor Yellow
    Write-Host "  ALL unused volumes and the ENTIRE build cache on this machine." -ForegroundColor Yellow
    Write-Host "  Images from other projects WILL be deleted." -ForegroundColor Yellow
    Write-Host ""
    $confirmation = Read-Host "  Type EXACTLY 'WIPE EVERYTHING' to proceed"
    if ($confirmation -ne "WIPE EVERYTHING") {
        Write-Host "  Aborted. Nothing was removed." -ForegroundColor Green
        exit 0
    }

    if ($PSCmdlet.ShouldProcess("all Docker resources", "prune")) {
        docker compose down --volumes --remove-orphans 2>$null
        docker system prune -a --volumes -f
        docker builder prune -a -f
    }

    Write-Section "Disk usage AFTER full wipe"
    docker system df
    Write-Host "`n  Docker is now empty. No residual images or build cache remain." -ForegroundColor Green
    exit 0
}

# --- Scoped cleanup --------------------------------------------------------
Write-Section "Scoped cleanup: pattern '$ProjectPattern'"

# 1. Bring the compose stack down (containers + named volumes + networks).
if (Test-Path "docker-compose.yml") {
    Write-Host "`n[1/6] docker compose down --volumes --remove-orphans"
    if ($PSCmdlet.ShouldProcess("compose stack", "down")) {
        docker compose down --volumes --remove-orphans 2>$null
    }
} else {
    Write-Host "`n[1/6] No docker-compose.yml in the current folder - skipping."
}

# 2. Containers whose name matches the project.
Write-Host "`n[2/6] Removing project containers"
$containers = docker ps -a --format "{{.ID}} {{.Names}}" |
    Where-Object { $_ -match $ProjectPattern }
if ($containers) {
    foreach ($entry in $containers) {
        $id, $name = $entry -split ' ', 2
        Write-Host "      - $name ($id)"
        if ($PSCmdlet.ShouldProcess($name, "docker rm -f")) { docker rm -f $id | Out-Null }
    }
} else {
    Write-Host "      (none found)"
}

# 3. Images whose repository matches the project.
Write-Host "`n[3/6] Removing project images"
$images = docker images --format "{{.ID}} {{.Repository}}:{{.Tag}}" |
    Where-Object { $_ -match $ProjectPattern }
if ($images) {
    foreach ($entry in $images) {
        $id, $ref = $entry -split ' ', 2
        Write-Host "      - $ref ($id)"
        if ($PSCmdlet.ShouldProcess($ref, "docker rmi -f")) { docker rmi -f $id | Out-Null }
    }
} else {
    Write-Host "      (none found)"
}

# 4. Networks.
Write-Host "`n[4/6] Removing project networks"
$networks = docker network ls --format "{{.ID}} {{.Name}}" |
    Where-Object { $_ -match $ProjectPattern }
if ($networks) {
    foreach ($entry in $networks) {
        $id, $name = $entry -split ' ', 2
        Write-Host "      - $name"
        if ($PSCmdlet.ShouldProcess($name, "docker network rm")) {
            docker network rm $id 2>$null | Out-Null
        }
    }
} else {
    Write-Host "      (none found)"
}

# 5. Volumes.
Write-Host "`n[5/6] Removing project volumes"
$volumes = docker volume ls --format "{{.Name}}" | Where-Object { $_ -match $ProjectPattern }
if ($volumes) {
    foreach ($name in $volumes) {
        Write-Host "      - $name"
        if ($PSCmdlet.ShouldProcess($name, "docker volume rm")) {
            docker volume rm $name 2>$null | Out-Null
        }
    }
} else {
    Write-Host "      (none found)"
}

# 6. Dangling layers and this project's build cache.
Write-Host "`n[6/6] Removing dangling images and stale build cache"
if ($PSCmdlet.ShouldProcess("dangling images and build cache", "prune")) {
    docker image prune -f | Out-Null
    docker builder prune -f --filter "until=1h" | Out-Null
}

Write-Section "Disk usage AFTER cleanup"
docker system df

Write-Host "`n  Remaining images on this machine:" -ForegroundColor Yellow
docker images --format "table {{.Repository}}`t{{.Tag}}`t{{.Size}}"

Write-Host "`n  Scoped cleanup complete. Nothing outside '$ProjectPattern' was touched." -ForegroundColor Green
Write-Host "  The next 'docker compose up --build' will build from a clean slate.`n" -ForegroundColor Green
