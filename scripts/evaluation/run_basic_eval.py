from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import html
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import httpx


DEFAULT_CASES = Path("evals/basic_cases.draft.jsonl")
DEFAULT_FORBIDDEN = (
    "DSH_SYSTEM_PROMPT",
    "KB_API_BASE",
    "minio://",
    "data/processing",
    "harness_sessions",
    "postgresql://",
)


@dataclass
class CaseResult:
    case_id: str
    category: str
    question: str
    review_criteria: str
    passed: bool
    latency_ms: float
    keyword_recall: float
    violations: list[str]
    answer: str
    error: str = ""


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: case must be an object")
        case_id = str(value.get("id") or "").strip()
        question = str(value.get("question") or "").strip()
        if not case_id or not question or case_id in seen:
            raise ValueError(f"line {line_number}: id/question must be non-empty and id unique")
        seen.add(case_id)
        cases.append(value)
    if not cases:
        raise ValueError("evaluation set is empty")
    return cases


def score_answer(case: dict[str, Any], answer: str, latency_ms: float) -> CaseResult:
    lowered = answer.lower()
    keywords = [str(item).strip() for item in case.get("expected_keywords", []) if str(item).strip()]
    matched = sum(1 for keyword in keywords if keyword.lower() in lowered)
    recall = matched / len(keywords) if keywords else 1.0
    forbidden = [*DEFAULT_FORBIDDEN, *(str(item) for item in case.get("forbidden_patterns", []))]
    violations = [pattern for pattern in forbidden if pattern and pattern.lower() in lowered]
    if not answer.strip():
        violations.append("empty_answer")
    minimum_recall = float(case.get("minimum_keyword_recall", 0.6 if keywords else 0.0))
    passed = bool(answer.strip()) and recall >= minimum_recall and not violations
    return CaseResult(
        case_id=str(case["id"]),
        category=str(case.get("category") or "uncategorized"),
        question=str(case.get("question") or ""),
        review_criteria=str(case.get("review_criteria") or ""),
        passed=passed,
        latency_ms=latency_ms,
        keyword_recall=recall,
        violations=violations,
        answer=answer,
    )


def run_live(cases: list[dict[str, Any]], base_url: str, timeout: float) -> list[CaseResult]:
    results: list[CaseResult] = []
    run_id = uuid4().hex
    with httpx.Client(timeout=timeout) as client:
        for case in cases:
            started = time.perf_counter()
            try:
                response = client.post(
                    f"{base_url.rstrip('/')}/query",
                    json={
                        "question": case["question"],
                        "user_id": f"eval:{run_id}",
                        "session_id": f"eval:{run_id}:{case['id']}",
                        "metadata": {"source": "basic-eval", "case_id": case["id"]},
                    },
                )
                response.raise_for_status()
                answer = str(response.json().get("answer") or "")
                result = score_answer(case, answer, (time.perf_counter() - started) * 1000)
            except Exception as exc:  # noqa: BLE001 - the report must retain every failed case.
                result = CaseResult(
                    case_id=str(case["id"]),
                    category=str(case.get("category") or "uncategorized"),
                    question=str(case.get("question") or ""),
                    review_criteria=str(case.get("review_criteria") or ""),
                    passed=False,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    keyword_recall=0.0,
                    violations=[],
                    answer="",
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
    return results


def write_report(results: list[CaseResult], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"basic-eval-{stamp}.json"
    html_path = output_dir / f"basic-eval-{stamp}.html"
    passed = sum(result.passed for result in results)
    payload = {
        "created_at": datetime.now().isoformat(),
        "summary": {"passed": passed, "total": len(results), "pass_rate": passed / len(results)},
        "results": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(result.case_id)}</td>"
        f"<td>{html.escape(result.category)}</td>"
        f"<td>{html.escape(result.question)}</td>"
        f"<td>{html.escape(result.answer)}</td>"
        f"<td>{html.escape(result.review_criteria)}</td>"
        f"<td>{'PASS' if result.passed else 'FAIL'}</td>"
        f"<td>{result.latency_ms:.0f}</td>"
        f"<td>{result.keyword_recall:.0%}</td>"
        f"<td>{html.escape(', '.join(result.violations) or result.error)}</td>"
        "</tr>"
        for result in results
    )
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>MTSCO Basic Eval</title>"
        "<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ddd;padding:.5rem;text-align:left;vertical-align:top}"
        "td:nth-child(4){white-space:pre-wrap;min-width:24rem}</style>"
        f"<h1>MTSCO Basic Eval</h1><p>{passed}/{len(results)} automatic checks passed. "
        "Draft cases still require an internal reviewer.</p>"
        "<table><thead><tr><th>ID</th><th>Category</th><th>Question</th><th>Answer</th>"
        "<th>Manual review criteria</th><th>Automatic result</th><th>Latency ms</th>"
        f"<th>Keyword recall</th><th>Violations/Error</th></tr></thead><tbody>{rows}</tbody></table>",
        encoding="utf-8",
    )
    return json_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal MTSCO knowledge-base evaluation.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/prod/api/v1")
    parser.add_argument("--output-dir", type=Path, default=Path("evals/results"))
    parser.add_argument("--timeout", type=float, default=660)
    parser.add_argument("--live", action="store_true", help="Actually call Harness; otherwise validate cases only.")
    args = parser.parse_args()
    cases = load_cases(args.cases)
    if not args.live:
        print(json.dumps({"valid": True, "cases": len(cases), "live": False}, ensure_ascii=False))
        return
    results = run_live(cases, args.base_url, args.timeout)
    json_path, html_path = write_report(results, args.output_dir)
    print(json.dumps({"json": str(json_path), "html": str(html_path)}, ensure_ascii=False))
    raise SystemExit(0 if all(result.passed for result in results) else 1)


if __name__ == "__main__":
    main()
