from __future__ import annotations

from collections import Counter
import threading


_lock = threading.Lock()
_http_requests: Counter[tuple[str, str, int]] = Counter()
_http_duration_seconds: Counter[tuple[str, str]] = Counter()


def observe_http(*, method: str, path: str, status: int, duration_seconds: float) -> None:
    # Route templates are not available in middleware before dispatch. Collapse
    # UUID-like/high-cardinality suffixes by keeping only the first four parts.
    normalized = "/".join(path.split("/")[:5]) or "/"
    with _lock:
        _http_requests[(method, normalized, status)] += 1
        _http_duration_seconds[(method, normalized)] += max(0.0, duration_seconds)


def render_prometheus(
    *, answer_jobs: dict, harness_sessions: dict[str, int]
) -> str:
    lines = [
        "# HELP mtsco_http_requests_total HTTP requests observed by this process.",
        "# TYPE mtsco_http_requests_total counter",
    ]
    with _lock:
        request_items = list(_http_requests.items())
        duration_items = list(_http_duration_seconds.items())
    for (method, path, status), value in request_items:
        lines.append(
            f'mtsco_http_requests_total{{method="{_escape(method)}",path="{_escape(path)}",status="{status}"}} {value}'
        )
    lines.extend([
        "# HELP mtsco_http_request_duration_seconds_total Accumulated HTTP request time in this process.",
        "# TYPE mtsco_http_request_duration_seconds_total counter",
    ])
    for (method, path), value in duration_items:
        lines.append(
            f'mtsco_http_request_duration_seconds_total{{method="{_escape(method)}",path="{_escape(path)}"}} {value:.6f}'
        )
    lines.extend([
        "# HELP mtsco_answer_jobs Durable Feishu answer jobs by status.",
        "# TYPE mtsco_answer_jobs gauge",
    ])
    for status, value in sorted((answer_jobs.get("counts") or {}).items()):
        lines.append(f'mtsco_answer_jobs{{status="{_escape(str(status))}"}} {int(value)}')
    lines.append(
        f'mtsco_answer_job_oldest_queued_seconds {float(answer_jobs.get("oldest_queued_seconds") or 0):.3f}'
    )
    lines.extend([
        "# HELP mtsco_harness_sessions Harness sessions by lifecycle state.",
        "# TYPE mtsco_harness_sessions gauge",
    ])
    for status, value in sorted(harness_sessions.items()):
        lines.append(f'mtsco_harness_sessions{{status="{_escape(status)}"}} {int(value)}')
    return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
