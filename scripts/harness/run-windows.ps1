# Starts the API on Windows with Harness enabled.  Start the database/vector
# services first, then point Feishu's public reverse proxy to this process.
[CmdletBinding()]
param(
    [string]$HarnessRoot = "C:\opt\mtsco\deepseek-harness",
    [int]$Port = 8000,
    [string]$ApiPrefix = "/prod"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
# Match python-dotenv's practical behaviour for the variables needed before
# Uvicorn starts.  Do not print values: this file commonly contains secrets.
$envFile = Join-Path $ProjectRoot ".env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content -LiteralPath $envFile -Encoding UTF8) {
        $text = $line.Trim()
        if (-not $text -or $text.StartsWith("#")) { continue }
        $match = [regex]::Match($text, '^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$')
        if (-not $match.Success) { continue }
        $key = $match.Groups[1].Value
        $value = $match.Groups[2].Value.Trim()
        if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        if (-not [Environment]::GetEnvironmentVariable($key, 'Process')) {
            [Environment]::SetEnvironmentVariable($key, $value, 'Process')
        }
    }
}
if (-not (Test-Path (Join-Path $HarnessRoot "packages\examples\jsonrpc-demo\lib\bin.js"))) {
    throw "Harness runtime is missing. Run .\scripts\harness\install-windows.ps1 first."
}
if (-not $env:DEEPSEEK_API_KEY) {
    throw "DEEPSEEK_API_KEY must be set in this PowerShell session or .env."
}

$env:DHS_REPO = $HarnessRoot
$env:HARNESS_ENABLED = "true"
$env:FEISHU_DURABLE_QUEUE_ENABLED = "false"
$env:HARNESS_MODEL = "deepseek-v4-flash"
$env:HARNESS_WORKDIR = (Join-Path $ProjectRoot "data\processing")
$env:HARNESS_SESSION_ROOT = (Join-Path $ProjectRoot "data\harness_sessions")
$normalizedPrefix = $ApiPrefix.Trim("/")
$env:KB_API_BASE = "http://127.0.0.1:$Port" + $(if ($normalizedPrefix) { "/$normalizedPrefix" } else { "" }) + "/api/v1"

Push-Location $ProjectRoot
try {
    & $pythonPath -m uvicorn app.main:app --host 0.0.0.0 --port $Port
} finally { Pop-Location }
