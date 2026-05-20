from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.services.llm import LLMAPIError, LLMClient


DEFAULT_DATASET_FILE = Path("data") / "dataset" / "test.json"
DEFAULT_OUTPUT_FILE = Path("data") / "dataset" / "evaluation.json"
DEFAULT_QUERY_URL = "http://localhost:8000/query"
DEFAULT_RETRIEVAL_PATH = "/retrieval/flow"
DEFAULT_RETRIEVAL_LIMIT = 15


@dataclass(frozen=True)
class QueryCallResult:
    answer: str
    raw_response: Any


@dataclass(frozen=True)
class RecallResult:
    score: int
    reason: str
    matched_chunk_index: int | None
    chunk_count: int
    latency_ms: float
    error: str = ""


def log(message: str, *, verbose: bool = True) -> None:
    if verbose:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [evaluate] {message}", flush=True)


JUDGE_SYSTEM_PROMPT = """你是企业内部知识库回答质量评测员。你需要基于题目、标准答案、文档依据和系统实际回答，给出严格但实用的评分。

评分维度：
1. readability 可读性：站在普通员工视角，预测用户愿意完整读完回答的概率。回答应清楚、简洁、有结构。图片等内容目前可能仍是占位符，不应因占位符本身扣太多分，但如果回答把占位符当成最终内容则应扣分。
2. correctness 正确性：回答是否与参考答案和文档依据一致，是否存在幻觉，是否纠正了问题里的错误前提。
3. completeness 完整性：回答是否覆盖参考答案的核心内容；回答多于参考答案但不矛盾，可以视为完整。

输出必须是 JSON，不要输出 Markdown 代码块。"""


JUDGE_USER_PROMPT_TEMPLATE = """请评测下面这条企业知识库问答结果。

题目类型：{question_types}
用户问题：
{question}

标准答案：
{reference_answer}

文档依据：
{evidence}

系统实际回答：
{actual_answer}

请输出 JSON 对象，字段如下：
- scores: 对象，包含 readability、correctness、completeness 三个 0-5 分数字。
- pass: 布尔值。若 correctness >= 4 且 completeness >= 4 且 readability >= 3，通常为 true。
- reasons: 对象，分别用一句中文解释 readability、correctness、completeness 的评分。
- missing_points: 字符串数组，列出遗漏的关键点；没有则为空数组。
- hallucinations: 字符串数组，列出与文档或标准答案冲突的内容；没有则为空数组。
- suggested_answer: 如果实际回答不合格，给出更好的简短答案；合格时可以为空字符串。
"""


def evaluate_dataset(
    dataset_file: str | Path = DEFAULT_DATASET_FILE,
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
    *,
    query_url: str = DEFAULT_QUERY_URL,
    retrieval_url: str | None = None,
    retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT,
    model: str | None = None,
    limit: int | None = None,
    seed: int | None = None,
    sleep_seconds: float = 0.0,
    user_id: str = "evaluation",
    verbose: bool = True,
) -> dict[str, Any]:
    dataset_path = Path(dataset_file)
    log(f"loading dataset: {dataset_path}", verbose=verbose)
    dataset = load_dataset(dataset_path)
    if limit is not None:
        original_count = len(dataset)
        dataset = random_sample_dataset(dataset, limit, seed=seed)
        log(
            f"random sampled {len(dataset)} of {original_count} item(s)"
            + (f" with seed={seed}" if seed is not None else ""),
            verbose=verbose,
        )
    else:
        log(f"loaded {len(dataset)} dataset item(s)", verbose=verbose)

    retrieval_url = retrieval_url or default_retrieval_url(query_url)
    log(f"query url: {query_url}", verbose=verbose)
    log(f"retrieval url: {retrieval_url}", verbose=verbose)
    log("initializing judge LLM client", verbose=verbose)
    llm = LLMClient()
    log(f"judge model: {model or llm.settings.model}", verbose=verbose)
    results: list[dict[str, Any]] = []

    for index, item in enumerate(dataset, start=1):
        question = str(item.get("question", "")).strip()
        if not question:
            log(f"[{index}/{len(dataset)}] skipped empty question", verbose=verbose)
            continue

        log(
            f"[{index}/{len(dataset)}] querying: {shorten(question, 80)}",
            verbose=verbose,
        )
        started = time.perf_counter()
        actual_answer = ""
        raw_query_response: Any = None
        query_error = ""
        try:
            query_result = call_query_api(
                query_url,
                question,
                user_id=user_id,
                metadata={
                    "evaluation_item_id": item.get("id"),
                    "document_name": item.get("document_name"),
                    "question_types": item.get("question_types", []),
                },
                verbose=verbose,
            )
            actual_answer = query_result.answer
            raw_query_response = query_result.raw_response
        except Exception as exc:  # noqa: BLE001 - keep batch evaluation moving.
            query_error = str(exc)

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        if query_error:
            log(
                f"[{index}/{len(dataset)}] query failed in {latency_ms}ms: {query_error}",
                verbose=verbose,
            )
        else:
            log(
                f"[{index}/{len(dataset)}] query returned in {latency_ms}ms, answer_chars={len(actual_answer)}",
                verbose=verbose,
            )

        log(f"[{index}/{len(dataset)}] checking approximate recall", verbose=verbose)
        recall = evaluate_recall(
            item=item,
            question=question,
            retrieval_url=retrieval_url,
            retrieval_limit=retrieval_limit,
            raw_query_response=raw_query_response,
            verbose=verbose,
        )
        log(
            f"[{index}/{len(dataset)}] recall={recall.score}, chunks={recall.chunk_count}, "
            f"latency={recall.latency_ms}ms, reason={recall.reason}",
            verbose=verbose,
        )

        log(f"[{index}/{len(dataset)}] judging answer quality", verbose=verbose)
        judge_started = time.perf_counter()
        try:
            judgment = judge_answer_logged(
                llm,
                item=item,
                actual_answer=actual_answer,
                query_error=query_error,
                model=model,
                verbose=verbose,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch evaluation moving.
            judgment = failed_judgment(f"judge failed: {exc}")
            log(f"[{index}/{len(dataset)}] judge failed: {exc}", verbose=verbose)
        log(
            f"[{index}/{len(dataset)}] judge finished in "
            f"{round((time.perf_counter() - judge_started) * 1000, 2)}ms",
            verbose=verbose,
        )

        results.append(
            {
                "item_id": item.get("id"),
                "document_name": item.get("document_name"),
                "document_path": item.get("document_path"),
                "question_types": item.get("question_types", []),
                "question": question,
                "reference_answer": item.get("reference_answer", ""),
                "evidence": item.get("evidence", ""),
                "actual_answer": actual_answer,
                "query_error": query_error,
                "latency_ms": latency_ms,
                "recall": {
                    "score": recall.score,
                    "reason": recall.reason,
                    "matched_chunk_index": recall.matched_chunk_index,
                    "chunk_count": recall.chunk_count,
                    "latency_ms": recall.latency_ms,
                    "error": recall.error,
                },
                "judgment": judgment,
            }
        )

        if sleep_seconds > 0 and index < len(dataset):
            log(f"[{index}/{len(dataset)}] sleeping {sleep_seconds}s", verbose=verbose)
            time.sleep(sleep_seconds)

    log(f"building summary for {len(results)} result(s)", verbose=verbose)
    report = {
        "metadata": {
            "dataset_file": str(dataset_path),
            "query_url": query_url,
            "retrieval_url": retrieval_url,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "judge_model": model or llm.settings.model,
            "total": len(results),
        },
        "summary": summarize_results(results),
        "results": results,
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing evaluation report: {output_path}", verbose=verbose)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log("evaluation complete", verbose=verbose)
    return report


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of QA items.")
    return [item for item in data if isinstance(item, dict)]


def random_sample_dataset(
    dataset: list[dict[str, Any]],
    limit: int,
    *,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if limit >= len(dataset):
        return list(dataset)
    rng = random.Random(seed)
    return rng.sample(dataset, limit)


def call_query_api(
    query_url: str,
    question: str,
    *,
    user_id: str,
    metadata: dict[str, Any],
    verbose: bool = True,
) -> QueryCallResult:
    payload = {
        "question": question,
        "user_id": user_id,
        "metadata": metadata,
    }
    timeout = httpx.Timeout(timeout=180.0, connect=10.0)
    log("posting query request", verbose=verbose)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(query_url, json=payload)
        response.raise_for_status()

    data: Any
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type.lower():
        data = response.json()
    else:
        text = response.text.strip()
        return QueryCallResult(answer=text, raw_response=text)

    answer = extract_answer(data)
    if not answer:
        raise ValueError(f"Query API response did not contain an answer: {data}")
    return QueryCallResult(answer=answer, raw_response=data)


def extract_answer(data: Any) -> str:
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, list):
        for item in data:
            answer = extract_answer(item)
            if answer:
                return answer
        return ""
    if isinstance(data, dict):
        for key in ("answer", "output", "text", "message", "result", "data"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, (dict, list)):
                answer = extract_answer(value)
                if answer:
                    return answer
    return ""


def evaluate_recall(
    *,
    item: dict[str, Any],
    question: str,
    retrieval_url: str,
    retrieval_limit: int,
    raw_query_response: Any,
    verbose: bool = True,
) -> RecallResult:
    started = time.perf_counter()
    chunks = extract_chunks(raw_query_response)
    source = "query_response"
    error = ""

    if not chunks:
        try:
            chunks = call_retrieval_api(
                retrieval_url,
                question,
                limit=retrieval_limit,
                document_name=processing_document_name(item),
                verbose=verbose,
            )
            source = "retrieval_api"
        except Exception as exc:  # noqa: BLE001 - recall should not block answer judging.
            error = str(exc)

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    if error:
        return RecallResult(
            score=0,
            reason=f"recall check failed: {error}",
            matched_chunk_index=None,
            chunk_count=0,
            latency_ms=latency_ms,
            error=error,
        )

    score, reason, matched_index = approximate_recall_score(item, chunks, source=source)
    return RecallResult(
        score=score,
        reason=reason,
        matched_chunk_index=matched_index,
        chunk_count=len(chunks),
        latency_ms=latency_ms,
    )


def call_retrieval_api(
    retrieval_url: str,
    question: str,
    *,
    limit: int,
    document_name: str | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    payload = {
        "query": question,
        "limit": limit,
        "rerank": True,
    }
    if document_name:
        payload["document_name"] = document_name
    timeout = httpx.Timeout(timeout=180.0, connect=10.0)
    log(
        f"posting retrieval request, limit={limit}, document_name={document_name or '<auto>'}",
        verbose=verbose,
    )
    with httpx.Client(timeout=timeout) as client:
        response = client.post(retrieval_url, json=payload)
        response.raise_for_status()

    data = response.json()
    chunks = data.get("chunks", []) if isinstance(data, dict) else []
    return [chunk for chunk in chunks if isinstance(chunk, dict)]


def extract_chunks(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        for key in ("chunks", "contexts", "documents", "retrieved_chunks", "references"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        for value in data.values():
            nested = extract_chunks(value)
            if nested:
                return nested
    if isinstance(data, list):
        for item in data:
            nested = extract_chunks(item)
            if nested:
                return nested
    return []


def approximate_recall_score(
    item: dict[str, Any],
    chunks: list[dict[str, Any]],
    *,
    source: str,
) -> tuple[int, str, int | None]:
    if not chunks:
        return 0, f"no chunks available from {source}", None

    targets = recall_targets(item)
    evidence = str(item.get("evidence", "")).strip()
    reference_answer = str(item.get("reference_answer", "")).strip()

    for index, chunk in enumerate(chunks, start=1):
        chunk_text = normalize_text(json.dumps(chunk, ensure_ascii=False))
        for target in targets:
            if target and normalize_text(target) in chunk_text:
                return 1, f"matched document target '{target}' in chunk from {source}", index

        if evidence and fuzzy_text_overlap(evidence, chunk_text) >= 0.45:
            return 1, f"matched evidence overlap in chunk from {source}", index

        if reference_answer and fuzzy_text_overlap(reference_answer, chunk_text) >= 0.45:
            return 1, f"matched reference-answer overlap in chunk from {source}", index

    return 0, f"no chunk matched document/evidence/reference from {source}", None


def recall_targets(item: dict[str, Any]) -> list[str]:
    raw_values = [
        item.get("document_name"),
        item.get("document_path"),
        item.get("source_document_name"),
        item.get("source_document_path"),
    ]
    targets: list[str] = []
    for value in raw_values:
        if not value:
            continue
        text = str(value).strip()
        targets.append(text)
        stem = Path(text).stem
        if stem and stem != text:
            targets.append(stem)
    return [target for target in dict.fromkeys(targets) if len(normalize_text(target)) >= 2]


def processing_document_name(item: dict[str, Any]) -> str | None:
    for key in ("source_document_name", "document_name", "source_document_path", "document_path"):
        value = item.get(key)
        if not value:
            continue
        stem = Path(str(value)).stem.strip()
        if stem:
            return stem
    return None


def fuzzy_text_overlap(reference: str, candidate: str) -> float:
    reference_tokens = text_tokens(reference)
    if not reference_tokens:
        return 0.0
    candidate_tokens = text_tokens(candidate)
    if not candidate_tokens:
        return 0.0
    matched = sum(1 for token in reference_tokens if token in candidate_tokens)
    return matched / len(reference_tokens)


def text_tokens(text: str) -> set[str]:
    normalized = normalize_text(text)
    ascii_tokens = re.findall(r"[a-zA-Z0-9_]{2,}", normalized)
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
    cjk_bigrams: list[str] = []
    for token in cjk_tokens:
        cjk_bigrams.extend(token[index : index + 2] for index in range(len(token) - 1))
    return set(ascii_tokens + cjk_bigrams)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def default_retrieval_url(query_url: str) -> str:
    if query_url.rstrip("/").endswith("/query"):
        return f"{query_url.rstrip('/')[:-len('/query')]}{DEFAULT_RETRIEVAL_PATH}"
    return f"{query_url.rstrip('/')}{DEFAULT_RETRIEVAL_PATH}"


def shorten(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def judge_answer_logged(
    llm: LLMClient,
    *,
    item: dict[str, Any],
    actual_answer: str,
    query_error: str,
    model: str | None,
    verbose: bool = True,
) -> dict[str, Any]:
    if query_error:
        return failed_judgment("query api failed")

    prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        question_types=", ".join(item.get("question_types", [])),
        question=item.get("question", ""),
        reference_answer=item.get("reference_answer", ""),
        evidence=item.get("evidence", ""),
        actual_answer=actual_answer,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        log("sending judge LLM request with JSON response_format", verbose=verbose)
        reply = llm.chat(
            messages,
            model=model,
            temperature=0.0,
            max_tokens=2500,
            extra_body={"response_format": {"type": "json_object"}},
        )
    except LLMAPIError as exc:
        if not looks_like_response_format_error(exc):
            log(f"judge LLM request failed before JSON fallback: {exc}", verbose=verbose)
            raise
        log(
            f"judge JSON-mode request failed, retrying without response_format: {exc}",
            verbose=verbose,
        )
        reply = llm.chat(
            messages,
            model=model,
            temperature=0.0,
            max_tokens=2500,
        )
    data = parse_json_object(reply)
    return normalize_judgment(data)


def failed_judgment(reason: str) -> dict[str, Any]:
    return {
        "scores": {
            "readability": 0,
            "correctness": 0,
            "completeness": 0,
        },
        "pass": False,
        "reasons": {
            "readability": reason,
            "correctness": reason,
            "completeness": reason,
        },
        "missing_points": [reason],
        "hallucinations": [],
        "suggested_answer": "",
    }


def looks_like_response_format_error(exc: LLMAPIError) -> bool:
    message = str(exc).lower()
    return (
        "response_format" in message
        or "json_object" in message
        or "json mode" in message
        or "returned 400" in message
        or "returned 422" in message
    )


def judge_answer(
    llm: LLMClient,
    *,
    item: dict[str, Any],
    actual_answer: str,
    query_error: str,
    model: str | None,
) -> dict[str, Any]:
    if query_error:
        return {
            "scores": {
                "readability": 0,
                "correctness": 0,
                "completeness": 0,
            },
            "pass": False,
            "reasons": {
                "readability": "查询接口调用失败，用户无法获得可读答案。",
                "correctness": "查询接口调用失败，无法验证答案正确性。",
                "completeness": "查询接口调用失败，未覆盖标准答案。",
            },
            "missing_points": ["查询接口调用失败"],
            "hallucinations": [],
            "suggested_answer": "",
        }

    prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        question_types=", ".join(item.get("question_types", [])),
        question=item.get("question", ""),
        reference_answer=item.get("reference_answer", ""),
        evidence=item.get("evidence", ""),
        actual_answer=actual_answer,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        reply = llm.chat(
            messages,
            model=model,
            temperature=0.0,
            max_tokens=2500,
            extra_body={"response_format": {"type": "json_object"}},
        )
    except LLMAPIError:
        reply = llm.chat(
            messages,
            model=model,
            temperature=0.0,
            max_tokens=2500,
        )
    data = parse_json_object(reply)
    return normalize_judgment(data)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match is None:
            raise
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("Judge response JSON must be an object.")
    return data


def normalize_judgment(data: dict[str, Any]) -> dict[str, Any]:
    raw_scores = data.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    scores = {
        "readability": clamp_score(raw_scores.get("readability")),
        "correctness": clamp_score(raw_scores.get("correctness")),
        "completeness": clamp_score(raw_scores.get("completeness")),
    }
    default_pass = (
        scores["readability"] >= 3
        and scores["correctness"] >= 4
        and scores["completeness"] >= 4
    )

    return {
        "scores": scores,
        "pass": bool(data.get("pass", default_pass)),
        "reasons": ensure_dict(data.get("reasons")),
        "missing_points": ensure_str_list(data.get("missing_points")),
        "hallucinations": ensure_str_list(data.get("hallucinations")),
        "suggested_answer": str(data.get("suggested_answer", "")).strip(),
    }


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(5.0, score))


def ensure_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(val) for key, val in value.items()}


def ensure_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "pass_rate": 0.0,
            "average_recall": 0.0,
            "average_scores": {
                "readability": 0.0,
                "correctness": 0.0,
                "completeness": 0.0,
            },
            "average_latency_ms": 0.0,
            "failed_query_count": 0,
        }

    score_names = ("readability", "correctness", "completeness")
    average_scores = {}
    for name in score_names:
        values = [
            result.get("judgment", {}).get("scores", {}).get(name, 0.0)
            for result in results
        ]
        average_scores[name] = round(statistics.mean(values), 2)

    passed = [result for result in results if result.get("judgment", {}).get("pass")]
    latencies = [float(result.get("latency_ms", 0.0)) for result in results]
    recall_scores = [
        int(result.get("recall", {}).get("score", 0))
        for result in results
    ]
    return {
        "pass_rate": round(len(passed) / len(results), 4),
        "average_recall": round(statistics.mean(recall_scores), 4),
        "recalled_count": sum(recall_scores),
        "passed_count": len(passed),
        "total_count": len(results),
        "average_scores": average_scores,
        "average_latency_ms": round(statistics.mean(latencies), 2),
        "failed_query_count": sum(1 for result in results if result.get("query_error")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FastAPI /query answers with a QA dataset.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_FILE),
        help=f"Dataset JSON file. Default: {DEFAULT_DATASET_FILE}",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help=f"Evaluation report output JSON file. Default: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--query-url",
        default=DEFAULT_QUERY_URL,
        help=f"FastAPI query endpoint. Default: {DEFAULT_QUERY_URL}",
    )
    parser.add_argument(
        "--retrieval-url",
        default=None,
        help="Retrieval endpoint for approximate recall. Defaults to query-url base + /retrieval/flow.",
    )
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=DEFAULT_RETRIEVAL_LIMIT,
        help="Number of chunks to request for approximate recall.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override judge LLM model name. Defaults to LLM_MODEL/SILICONFLOW_MODEL.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Randomly sample and evaluate N items.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used when --limit samples the dataset.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between query requests.",
    )
    parser.add_argument(
        "--user-id",
        default="evaluation",
        help="User id sent to the query API.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_dataset(
        args.dataset,
        args.output,
        query_url=args.query_url,
        retrieval_url=args.retrieval_url,
        retrieval_limit=args.retrieval_limit,
        model=args.model,
        limit=args.limit,
        seed=args.seed,
        sleep_seconds=args.sleep,
        user_id=args.user_id,
        verbose=not args.quiet,
    )
    summary = report["summary"]
    print(
        "Evaluation complete: "
        f"{summary.get('passed_count', 0)}/{summary.get('total_count', 0)} passed, "
        f"pass_rate={summary.get('pass_rate', 0.0)}, "
        f"average_recall={summary.get('average_recall', 0.0)}, "
        f"average_scores={summary.get('average_scores', {})}. "
        f"Wrote report to {args.output}"
    )


if __name__ == "__main__":
    main()
