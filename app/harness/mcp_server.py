"""stdio MCP bridge exposing only the approved KB capabilities."""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

# This process is launched by Node with the Harness checkout as cwd.  Add the
# repository root before importing application services below.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KB_API_BASE = os.getenv("KB_API_BASE", "http://127.0.0.1:8000/prod/api/v1").rstrip("/")
USER_ID = os.getenv("KB_USER_ID", "")

TOOLS = [
    {"name": "kb_hybrid_search", "description": "企业知识库混合检索。", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}},
    {"name": "kb_graph_search", "description": "产品和标准关系图谱检索。", "inputSchema": {"type": "object", "properties": {"keyword": {"type": "string"}, "port": {"enum": ["product-standards", "standard-context"]}}, "required": ["keyword", "port"]}},
    {"name": "memory_search", "description": "仅检索当前用户的已归档对话记忆。", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]


def _post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{KB_API_BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode())


def _result(value: object, error: bool = False) -> dict:
    return {"isError": error, "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value}]}


def call(name: str, args: dict) -> dict:
    try:
        if name == "kb_hybrid_search":
            return _result(_post("/retrieval/flow", {"query": args["query"], "limit": min(int(args.get("limit", 8)), 20), "rerank": True}))
        if name == "kb_graph_search":
            return _result(_post(f"/graph/{args['port']}", {"keyword": args["keyword"], "limit": 20}))
        if name == "memory_search":
            from app.db.postgres import list_harness_memories
            from app.db.minio import get_minio_client, parse_raw_document_reference
            memories = list_harness_memories(user_id=USER_ID, query=str(args["query"]))
            # Metadata filtering is done before object access and user_id is
            # supplied only by the trusted API process.  The model never gets
            # an object-store credential or an arbitrary object path input.
            for memory in memories:
                reference = parse_raw_document_reference(memory["object_uri"])
                response = get_minio_client().get_object(reference.bucket, reference.object_name)
                try:
                    memory["content"] = response.read(24_000).decode("utf-8", errors="replace")
                finally:
                    response.close()
                    response.release_conn()
            return _result(memories)
        return _result("未知工具", True)
    except Exception as exc:  # MCP failures must become model-visible tool errors.
        return _result(f"工具调用失败：{type(exc).__name__}: {exc}", True)


for line in sys.stdin:
    try:
        message = json.loads(line)
        if "id" not in message:
            continue
        method = message.get("method")
        if method == "initialize":
            payload = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "mtsco-kb", "version": "1"}}
        elif method == "tools/list":
            payload = {"tools": TOOLS}
        elif method == "tools/call":
            params = message.get("params") or {}
            payload = call(params.get("name", ""), params.get("arguments") or {})
        else:
            payload = {}
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": payload}, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except Exception:
        continue
