[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$env:HARNESS_IDLE_SECONDS = "25200"
# $env:HARNESS_IDLE_SECONDS = "300"
$env:HARNESS_ENABLED = "true"
Push-Location $ProjectRoot
try {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting Harness memory scheduler."
    Write-Host "  Idle cutoff: $($env:HARNESS_IDLE_SECONDS) seconds (7 hours)"
    Write-Host "  Scheduler poll interval: loaded from HARNESS_SCHEDULER_INTERVAL_SECONDS (default: 300 seconds)"
    Write-Host "  Length rollover: 90000 tokens by default; handoff summary is consumed once by the new session."
    Write-Host "  Summary model: configured by HARNESS_MEMORY_SUMMARY_MODEL; deterministic fallback is used on failure."
    Write-Host "  Metadata table: harness_memories; files: MinIO bucket configured by HARNESS_MEMORY_BUCKET."
    & $pythonPath -m app.workers.harness_scheduler
    if ($LASTEXITCODE -ne 0) { throw "Harness memory scheduler exited with code $LASTEXITCODE." }
} finally { Pop-Location }
