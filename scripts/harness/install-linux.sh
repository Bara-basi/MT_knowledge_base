#!/usr/bin/env bash
# Host installation is the supported first deployment mode.  It avoids nesting
# Node's Harness runtime inside the API image; Docker can later use the same
# checkout after this path has proven stable.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS_DIR="${DHS_REPO:-/opt/mtsco/deepseek-harness}"
HARNESS_REPOSITORY="${HARNESS_REPOSITORY:-https://github.com/deepseek-ai/deepseek-harness.git}"
HARNESS_REF="${HARNESS_REF:-dsh-v0.1.0-rc.7}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "$HARNESS_DIR/.git" ]]; then
  mkdir -p "$(dirname "$HARNESS_DIR")"
  git clone --depth 1 --branch "$HARNESS_REF" "$HARNESS_REPOSITORY" "$HARNESS_DIR"
fi
git -C "$HARNESS_DIR" fetch --depth 1 origin "$HARNESS_REF" || true
git -C "$HARNESS_DIR" checkout --detach "$HARNESS_REF"
for PATCH_FILE in \
  "$PROJECT_ROOT/app/harness/patches/dsh-sdk-jsonrpc-session-resume.patch" \
  "$PROJECT_ROOT/app/harness/patches/rooted-readonly-fs.patch"; do
  if git -C "$HARNESS_DIR" apply --recount --check "$PATCH_FILE"; then
    git -C "$HARNESS_DIR" apply --recount "$PATCH_FILE"
  elif [[ "$(basename "$PATCH_FILE")" == "rooted-readonly-fs.patch" ]] \
    && grep -q "rooted: z.boolean().default(false)" "$HARNESS_DIR/packages/fs/fs-local/src/index.ts" \
    && grep -q "readOnly: z.boolean().default(false)" "$HARNESS_DIR/packages/fs/tool-fs/src/index.ts"; then
    : # already applied; reverse checking multi-file generated patches is brittle on Windows line endings
  elif ! git -C "$HARNESS_DIR" apply --recount --reverse --check "$PATCH_FILE"; then
    echo "Harness patch does not match the pinned source revision: $PATCH_FILE" >&2
    exit 1
  fi
done
corepack enable
( cd "$HARNESS_DIR" && pnpm install --frozen-lockfile && pnpm run build )

if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$PROJECT_ROOT/.venv"
fi
"$PROJECT_ROOT/.venv/bin/python" -m pip install -e "$HARNESS_DIR/python/sdk-runtime" -e "$HARNESS_DIR/python/sdk"

mkdir -p "$PROJECT_ROOT/app/harness/node_modules/@deepseek-ai"
for pair in \
  'dsh-sdk-jsonrpc-server:packages/sdk/server' 'dsh-llm-pi-ai:packages/llm/llm-pi-ai' 'dsh-llm-retry:packages/llm/llm-retry' \
  'dsh-agent-spine-demo:packages/examples/agent-spine-demo' 'dsh-subprocess-local:packages/subprocess/subprocess-local' \
  'dsh-fs-local:packages/fs/fs-local' 'dsh-fs-observation-policy:packages/fs/fs-observation-policy' \
  'dsh-tool-fs:packages/fs/tool-fs' 'dsh-tool-fs-search:packages/fs/tool-fs-search' \
  'dsh-web:packages/web/web' 'dsh-web-search-deepseek:packages/web/web-search-deepseek' 'dsh-web-fetch-http:packages/web/web-fetch-http' 'dsh-tool-web:packages/web/tool-web' \
  'dsh-session-persistence-jsonl:packages/session/session-persistence-jsonl' \
  'dsh-compaction-basic:packages/compaction/compaction-basic' 'dsh-token-meter:packages/llm/token-meter' \
  'dsh-mcp-client:packages/mcp/mcp-client' 'cordis:vendor/cordis' 'cosmokit:vendor/cosmokit' 'schemastery:vendor/schemastery'; do
  name="${pair%%:*}"; source_path="${pair#*:}"
  ln -sfn "$HARNESS_DIR/$source_path" "$PROJECT_ROOT/app/harness/node_modules/@deepseek-ai/$name"
done

echo 'Harness installed. Set HARNESS_ENABLED=true only after API smoke tests pass.'
