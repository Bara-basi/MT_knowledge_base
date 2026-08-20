# Installs a pinned source-mode DeepSeek Harness for local Windows testing.
# Run from the repository root in an elevated-free PowerShell session.
[CmdletBinding()]
param(
    [string]$HarnessRoot = "C:\opt\mtsco\deepseek-harness",
    [string]$HarnessRepository = "https://github.com/deepseek-ai/deepseek-harness.git",
    [string]$HarnessRef = "dsh-v0.1.0-rc.7",
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pythonPath = Join-Path $ProjectRoot $Python
if (-not (Test-Path $pythonPath)) {
    py -3.11 -m venv (Join-Path $ProjectRoot ".venv")
}

if (Test-Path (Join-Path $HarnessRoot ".git")) {
    git -C $HarnessRoot fetch --depth 1 origin $HarnessRef
} else {
    New-Item -ItemType Directory -Force -Path (Split-Path $HarnessRoot) | Out-Null
    git clone --depth 1 --branch $HarnessRef $HarnessRepository $HarnessRoot
}
git -C $HarnessRoot checkout --detach $HarnessRef
$patchFile = Join-Path $ProjectRoot "app\harness\patches\dsh-sdk-jsonrpc-session-resume.patch"
git -C $HarnessRoot apply --recount --check $patchFile 2>$null
if ($LASTEXITCODE -eq 0) {
    git -C $HarnessRoot apply --recount $patchFile
} else {
    git -C $HarnessRoot apply --recount --reverse --check $patchFile 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Harness session-resume patch does not match the pinned source revision."
    }
}
corepack enable
Push-Location $HarnessRoot
try {
    pnpm install --frozen-lockfile
    pnpm run build
} finally { Pop-Location }

& $pythonPath -m pip install -e (Join-Path $HarnessRoot "python\sdk-runtime") -e (Join-Path $HarnessRoot "python\sdk")

$nodeModules = Join-Path $ProjectRoot "app\harness\node_modules\@deepseek-ai"
New-Item -ItemType Directory -Force -Path $nodeModules | Out-Null
$links = @{
    "dsh-sdk-jsonrpc-server" = "packages\sdk\server";
    "dsh-llm-pi-ai" = "packages\llm\llm-pi-ai";
    "dsh-agent-spine-demo" = "packages\examples\agent-spine-demo";
    "dsh-subprocess-local" = "packages\subprocess\subprocess-local";
    "dsh-fs-local" = "packages\fs\fs-local";
    "dsh-fs-observation-policy" = "packages\fs\fs-observation-policy";
    "dsh-tool-fs" = "packages\fs\tool-fs";
    "dsh-tool-fs-search" = "packages\fs\tool-fs-search";
    "dsh-web" = "packages\web\web";
    "dsh-web-search-deepseek" = "packages\web\web-search-deepseek";
    "dsh-web-fetch-http" = "packages\web\web-fetch-http";
    "dsh-tool-web" = "packages\web\tool-web";
    "dsh-session-persistence-jsonl" = "packages\session\session-persistence-jsonl";
    "dsh-compaction-basic" = "packages\compaction\compaction-basic";
    "dsh-token-meter" = "packages\llm\token-meter";
    "dsh-mcp-client" = "packages\mcp\mcp-client";
    "cordis" = "vendor\cordis"; "cosmokit" = "vendor\cosmokit"; "schemastery" = "vendor\schemastery";
}
foreach ($name in $links.Keys) {
    $target = Join-Path $HarnessRoot $links[$name]
    $link = Join-Path $nodeModules $name
    if (Test-Path $link) { Remove-Item -LiteralPath $link -Force -Recurse }
    New-Item -ItemType SymbolicLink -Path $link -Target $target | Out-Null
}

Write-Host "Harness runtime installed at $HarnessRoot"
Write-Host "Next: .\scripts\harness\run-windows.ps1 -HarnessRoot '$HarnessRoot'"
