from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


CHECKPOINTS = [
    {
        "name": "fastapi_query_entry",
        "signal": "client request start offsets and HTTP completion spread",
        "risk": "FastAPI worker/event-loop or inbound connection backlog",
    },
    {
        "name": "n8n_execution_start",
        "signal": "n8n execution startedAt spread within each batch",
        "risk": "n8n webhook or workflow execution queue",
    },
    {
        "name": "retrieval_subworkflow",
        "signal": "Execute Workflow node durations such as 前往问答检索*",
        "risk": "FastAPI retrieval endpoint, Milvus/BM25/reranker, local model contention",
    },
    {
        "name": "final_answer_model",
        "signal": "检索问答 and its chat model node executionTime",
        "risk": "LLM provider latency, rate limits, n8n model node concurrency",
    },
    {
        "name": "fastapi_final_processing",
        "signal": "client duration minus matched n8n execution duration",
        "risk": "response parsing, Feishu card/reference processing, outbound Feishu API",
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run same-question concurrent QA probes and collect n8n node timings.",
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("BENCH_DATASET", "data/dataset/test_0611.json"),
        help="Dataset JSON path. The script samples question fields from this file.",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("BENCH_QUERY_URL", "http://localhost:8000/api/v1/query"),
        help="FastAPI query endpoint.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("BENCH_CONCURRENCY", "12")),
        help="Concurrent requests per batch.",
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=int(os.getenv("BENCH_BATCH_COUNT", "2")),
        help="Number of same-question batches to run.",
    )
    parser.add_argument(
        "--question-offset",
        type=int,
        default=int(os.getenv("BENCH_QUESTION_OFFSET", "0")),
        help="Question index offset in the dataset.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("BENCH_REQUEST_TIMEOUT", "900")),
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--between-batch-sleep",
        type=float,
        default=float(os.getenv("BENCH_BETWEEN_BATCH_SLEEP", "8")),
        help="Sleep seconds between batches.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("BENCH_OUTPUT_DIR", "data/benchmark"),
        help="Directory for JSON reports.",
    )
    parser.add_argument(
        "--n8n-base-url",
        default=os.getenv("N8N_API_BASE_URL", "").rstrip("/"),
        help="n8n API base URL.",
    )
    parser.add_argument(
        "--n8n-api-key",
        default=os.getenv("N8N_API_KEY", ""),
        help="n8n API key. Prefer environment variables over command-line use.",
    )
    parser.add_argument(
        "--n8n-query-workflow-id",
        default=os.getenv("N8N_QUERY_WORKFLOW_ID", ""),
        help="n8n query workflow id.",
    )
    return parser.parse_args()


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("\"'")


def _parse_time(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return ordered[int(k)]
    return ordered[lower] * (upper - k) + ordered[upper] * (k - lower)


def _summarize_durations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [row["duration_s"] for row in rows if row.get("duration_s") is not None]
    if not durations:
        return {}
    split_at = max(1, len(durations) // 2)
    first_half = durations[:split_at]
    second_half = durations[split_at:]
    return {
        "count": len(durations),
        "min_s": round(min(durations), 3),
        "max_s": round(max(durations), 3),
        "mean_s": round(statistics.mean(durations), 3),
        "median_s": round(statistics.median(durations), 3),
        "p90_s": round(_percentile(durations, 0.9), 3),
        "first_half_mean_s": round(statistics.mean(first_half), 3) if first_half else None,
        "second_half_mean_s": round(statistics.mean(second_half), 3) if second_half else None,
        "tail_minus_head_mean_s": (
            round(statistics.mean(second_half) - statistics.mean(first_half), 3)
            if first_half and second_half
            else None
        ),
    }


def _node_summary(execution: dict[str, Any]) -> dict[str, Any]:
    data = execution.get("data") if isinstance(execution.get("data"), dict) else {}
    result = data.get("resultData") if isinstance(data.get("resultData"), dict) else {}
    run_data = result.get("runData") if isinstance(result.get("runData"), dict) else {}
    nodes: list[dict[str, Any]] = []
    for name, runs in run_data.items():
        if not isinstance(runs, list) or not runs or not isinstance(runs[0], dict):
            continue
        run = runs[0]
        nodes.append(
            {
                "name": name,
                "startTime": run.get("startTime"),
                "executionTime_ms": run.get("executionTime"),
            }
        )
    return {
        "lastNodeExecuted": result.get("lastNodeExecuted"),
        "nodes": nodes,
    }


async def _fetch_n8n_executions(
    *,
    n8n_base_url: str,
    n8n_api_key: str,
    workflow_id: str,
    started_after_ts: float,
    stopped_before_ts: float | None,
) -> list[dict[str, Any]]:
    if not n8n_base_url or not n8n_api_key or not workflow_id:
        return []

    headers = {"X-N8N-API-KEY": n8n_api_key}
    collected: dict[str, dict[str, Any]] = {}
    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        for status in (None, "running", "success", "error"):
            params: dict[str, Any] = {
                "workflowId": workflow_id,
                "limit": 100,
                "includeData": "false",
            }
            if status:
                params["status"] = status
            response = await client.get(f"{n8n_base_url}/api/v1/executions", params=params)
            response.raise_for_status()
            for item in response.json().get("data") or []:
                started = _parse_time(item.get("startedAt"))
                if started is None or started < started_after_ts - 3:
                    continue
                if stopped_before_ts is not None and started > stopped_before_ts + 10:
                    continue
                execution_id = str(item.get("id"))
                if execution_id:
                    collected[execution_id] = item

        details: list[dict[str, Any]] = []
        for execution_id, item in sorted(
            collected.items(),
            key=lambda pair: _parse_time(pair[1].get("startedAt")) or 0,
        ):
            response = await client.get(
                f"{n8n_base_url}/api/v1/executions/{execution_id}",
                params={"includeData": "true"},
            )
            response.raise_for_status()
            detail = response.json()
            detail["_list_status"] = item.get("status")
            detail["_list_finished"] = item.get("finished")
            detail["_node_summary"] = _node_summary(detail)
            details.append(detail)
    return details


async def _send_one(
    *,
    client: httpx.AsyncClient,
    api_url: str,
    question: str,
    batch_name: str,
    request_index: int,
    batch_wall_start: float,
    release: asyncio.Event,
) -> dict[str, Any]:
    await release.wait()
    start = time.perf_counter()
    wall_start = time.time()
    payload = {
        "question": question,
        "user_id": "concurrency-benchmark",
        "session_id": f"{batch_name}-session",
        "conversation_id": f"{batch_name}-conversation",
        "metadata": {
            "source": "concurrency_benchmark",
            "batch": batch_name,
            "request_index": request_index,
        },
    }
    try:
        response = await client.post(api_url, json=payload)
        duration = time.perf_counter() - start
        answer_len = None
        try:
            body = response.json()
            answer = body.get("answer") if isinstance(body, dict) else None
            answer_len = len(answer) if isinstance(answer, str) else None
        except ValueError:
            pass
        return {
            "batch": batch_name,
            "request_index": request_index,
            "start_offset_s": round(wall_start - batch_wall_start, 3),
            "wall_start": datetime.fromtimestamp(wall_start, timezone.utc).isoformat(),
            "status_code": response.status_code,
            "duration_s": round(duration, 3),
            "answer_len": answer_len,
            "body_preview": response.text[:300].replace("\n", "\\n"),
        }
    except Exception as exc:  # noqa: BLE001 - benchmark rows should capture failures.
        duration = time.perf_counter() - start
        return {
            "batch": batch_name,
            "request_index": request_index,
            "start_offset_s": round(wall_start - batch_wall_start, 3),
            "wall_start": datetime.fromtimestamp(wall_start, timezone.utc).isoformat(),
            "status_code": None,
            "duration_s": round(duration, 3),
            "error": repr(exc),
        }


async def _run_batch(args: argparse.Namespace, batch_name: str, question: str) -> dict[str, Any]:
    release = asyncio.Event()
    timeout = httpx.Timeout(
        timeout=args.timeout,
        connect=20,
        read=args.timeout,
        write=60,
    )
    limits = httpx.Limits(
        max_connections=args.concurrency + 5,
        max_keepalive_connections=args.concurrency + 5,
    )
    batch_wall_start = time.time()
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = [
            asyncio.create_task(
                _send_one(
                    client=client,
                    api_url=args.api_url,
                    question=question,
                    batch_name=batch_name,
                    request_index=index,
                    batch_wall_start=batch_wall_start,
                    release=release,
                )
            )
            for index in range(args.concurrency)
        ]
        await asyncio.sleep(0.2)
        release.set()
        rows = await asyncio.gather(*tasks)
    batch_wall_end = time.time()

    await asyncio.sleep(2)
    executions = await _fetch_n8n_executions(
        n8n_base_url=args.n8n_base_url,
        n8n_api_key=args.n8n_api_key,
        workflow_id=args.n8n_query_workflow_id,
        started_after_ts=batch_wall_start,
        stopped_before_ts=batch_wall_end,
    )

    requests = sorted(rows, key=lambda row: row["request_index"])
    return {
        "batch": batch_name,
        "question_preview": question[:200],
        "concurrency": args.concurrency,
        "wall_start": datetime.fromtimestamp(batch_wall_start, timezone.utc).isoformat(),
        "wall_end": datetime.fromtimestamp(batch_wall_end, timezone.utc).isoformat(),
        "wall_duration_s": round(batch_wall_end - batch_wall_start, 3),
        "summary": _summarize_durations(requests),
        "requests": requests,
        "n8n_executions": [_format_execution(item) for item in executions],
    }


def _format_execution(item: dict[str, Any]) -> dict[str, Any]:
    started = _parse_time(item.get("startedAt"))
    stopped = _parse_time(item.get("stoppedAt"))
    return {
        "id": item.get("id"),
        "status": item.get("status") or item.get("_list_status"),
        "finished": (
            item.get("finished")
            if item.get("finished") is not None
            else item.get("_list_finished")
        ),
        "startedAt": item.get("startedAt"),
        "stoppedAt": item.get("stoppedAt"),
        "duration_s": round((stopped or time.time()) - started, 3) if started else None,
        "lastNodeExecuted": item.get("_node_summary", {}).get("lastNodeExecuted"),
        "nodes": item.get("_node_summary", {}).get("nodes"),
    }


def _load_questions(dataset_path: Path) -> list[str]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [
        item["question"].strip()
        for item in data
        if isinstance(item, dict)
        and isinstance(item.get("question"), str)
        and item["question"].strip()
    ]


async def _main() -> None:
    _load_dotenv(Path(".env"))
    args = _parse_args()
    dataset_path = Path(args.dataset)
    questions = _load_questions(dataset_path)
    selected = questions[args.question_offset : args.question_offset + args.batch_count]
    if len(selected) < args.batch_count:
        raise RuntimeError(f"not enough question fields in {dataset_path}")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "n8n_base_url": args.n8n_base_url,
        "n8n_query_workflow_id": args.n8n_query_workflow_id,
        "concurrency": args.concurrency,
        "batch_count": args.batch_count,
        "request_timeout_s": args.timeout,
        "checkpoints": CHECKPOINTS,
        "batches": [],
    }

    for index, question in enumerate(selected, start=1):
        batch_name = f"batch_{index:02d}"
        print(
            f"RUN {batch_name} concurrency={args.concurrency} "
            f"question_preview={question[:80]!r}",
            flush=True,
        )
        batch = await _run_batch(args, batch_name, question)
        report["batches"].append(batch)
        print(f"DONE {batch_name} summary={batch['summary']}", flush=True)
        print(
            "REQUESTS "
            + json.dumps(
                [
                    {
                        key: row.get(key)
                        for key in (
                            "request_index",
                            "status_code",
                            "duration_s",
                            "answer_len",
                            "error",
                        )
                    }
                    for row in batch["requests"]
                ],
                ensure_ascii=False,
            ),
            flush=True,
        )
        print(
            "N8N "
            + json.dumps(
                [
                    {
                        key: execution.get(key)
                        for key in (
                            "id",
                            "status",
                            "finished",
                            "startedAt",
                            "stoppedAt",
                            "duration_s",
                            "lastNodeExecuted",
                        )
                    }
                    for execution in batch["n8n_executions"]
                ],
                ensure_ascii=False,
            ),
            flush=True,
        )
        if index < len(selected):
            await asyncio.sleep(args.between_batch_sleep)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"concurrency_probe_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REPORT {output_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(_main())
