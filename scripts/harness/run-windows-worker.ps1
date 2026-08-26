# Starts the durable Feishu answer worker with the same Harness runtime as API.
[CmdletBinding()]
param(
    [string]$HarnessRoot = "C:\opt\mtsco\deepseek-harness",
    [int]$Concurrency = 1
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$runtimeEntry = Join-Path $HarnessRoot "packages\examples\jsonrpc-demo\lib\bin.js"
if (-not (Test-Path $runtimeEntry)) {
    throw "Harness runtime is missing. Run .\scripts\harness\install-windows.ps1 first."
}

$env:DHS_REPO = $HarnessRoot
$env:HARNESS_ENABLED = "true"
$env:HARNESS_WORKDIR = (Join-Path $ProjectRoot "data\processing")
$env:HARNESS_SESSION_ROOT = (Join-Path $ProjectRoot "data\harness_sessions")
$env:FEISHU_DURABLE_QUEUE_ENABLED = "true"
$env:ANSWER_WORKER_CONCURRENCY = [string][Math]::Max(1, $Concurrency)

Push-Location $ProjectRoot
try {
    & $pythonPath -m app.workers.answer_worker
    if ($LASTEXITCODE -ne 0) { throw "Answer worker exited with code $LASTEXITCODE." }
} finally { Pop-Location }
