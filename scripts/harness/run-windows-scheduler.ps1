[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$env:HARNESS_ENABLED = "true"
Push-Location $ProjectRoot
try {
    & $pythonPath -m app.workers.harness_scheduler
} finally { Pop-Location }
