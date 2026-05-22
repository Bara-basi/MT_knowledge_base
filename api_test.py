import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests


EXCLUDE_KIND_RE = re.compile(
    r"(embed|embedding|rerank|bge|jina|gte|clip|siglip|colbert|retriev|moderation|"
    r"image|kolors|wan|voice|audio|speech|asr|tts|ocr|vl|captioner)",
    re.IGNORECASE,
)
EXCLUDE_VENDOR_RE = re.compile(r"(qwen|deepseek)", re.IGNORECASE)

SIMPLE_PROMPT = (
    "Return JSON only. What is 17 * 23? "
    'Use exactly this schema: {"answer": number, "brief_reason": string}.'
)
COMPLEX_PROMPT = (
    "Return JSON only. You are choosing an LLM for an internal enterprise knowledge base. "
    "Compare answer quality, latency, JSON reliability, and controllable reasoning. "
    'Use exactly this schema: {"answer": string, "risks": [string], "score": number}.'
)


@dataclass
class TestResult:
    model: str
    ok: bool
    excluded_qwen_deepseek: bool
    simple_ok: bool = False
    complex_ok: bool = False
    json_ok: bool = False
    thinking_off_ok: bool = False
    thinking_on_ok: bool = False
    verbosity_ok: bool = False
    simple_latency_s: Optional[float] = None
    complex_latency_s: Optional[float] = None
    simple_chars: int = 0
    complex_chars: int = 0
    simple_answer: str = ""
    complex_answer: str = ""
    errors: List[str] = None
    score_simple: float = 0.0
    score_complex: float = 0.0

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def load_env(path: str = ".env") -> Dict[str, str]:
    env: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def list_models(base_url: str, api_key: str) -> List[str]:
    resp = requests.get(f"{base_url.rstrip('/')}/models", headers=headers(api_key), timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", payload)
    models = []
    for item in data:
        if isinstance(item, dict):
            model_id = item.get("id") or item.get("model") or item.get("name")
        else:
            model_id = str(item)
        if model_id:
            models.append(model_id)
    return sorted(set(models))


def extract_content(payload: Dict[str, Any]) -> str:
    try:
        msg = payload["choices"][0]["message"]
        content = msg.get("content", "")
        if isinstance(content, list):
            return "".join(str(part.get("text", part)) for part in content)
        return content or ""
    except Exception:
        return ""


def parse_jsonish(text: str) -> Tuple[bool, Any]:
    text = text.strip()
    if not text:
        return False, None
    try:
        return True, json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return True, json.loads(match.group(0))
        except Exception:
            return False, None
    return False, None


def chat_once(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    *,
    enable_thinking: bool,
    verbosity: str,
    max_tokens: int,
) -> Tuple[bool, float, str, str]:
    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You must output valid JSON only. "
                    f"Reasoning verbosity target: {verbosity}. "
                    "Do not include markdown fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.8,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "stream": False,
        # SiliconFlow's OpenAI-compatible endpoint accepts extra model-specific
        # options for reasoning-capable models. Unsupported models may reject
        # these, which is part of this capability probe.
        "enable_thinking": enable_thinking,
        "thinking_budget": 1024 if enable_thinking else 0,
    }
    if verbosity == "low":
        body["frequency_penalty"] = 0.2

    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers(api_key),
            json=body,
            timeout=35,
        )
        latency = time.perf_counter() - start
        if resp.status_code >= 400:
            return False, latency, "", f"HTTP {resp.status_code}: {resp.text[:300]}"
        payload = resp.json()
        return True, latency, extract_content(payload), ""
    except Exception as exc:
        return False, time.perf_counter() - start, "", repr(exc)


def score_result(result: TestResult) -> None:
    if result.simple_ok and result.json_ok and result.simple_latency_s is not None:
        latency_score = max(0.0, 40.0 - result.simple_latency_s * 6.0)
        compact_score = 15.0 if result.simple_chars <= 220 else max(0.0, 15.0 - (result.simple_chars - 220) / 80)
        result.score_simple = latency_score + compact_score
        if result.thinking_off_ok and result.thinking_on_ok:
            result.score_simple += 20.0
        if result.verbosity_ok:
            result.score_simple += 10.0

    if result.complex_ok and result.json_ok and result.complex_latency_s is not None:
        latency_score = max(0.0, 30.0 - result.complex_latency_s * 2.0)
        depth_score = min(30.0, result.complex_chars / 25.0)
        result.score_complex = latency_score + depth_score
        if result.thinking_off_ok and result.thinking_on_ok:
            result.score_complex += 25.0
        if result.verbosity_ok:
            result.score_complex += 10.0


def test_model(base_url: str, api_key: str, model: str) -> TestResult:
    result = TestResult(
        model=model,
        ok=False,
        excluded_qwen_deepseek=bool(EXCLUDE_VENDOR_RE.search(model)),
    )

    ok1, lat1, text1, err1 = chat_once(
        base_url,
        api_key,
        model,
        SIMPLE_PROMPT,
        enable_thinking=False,
        verbosity="low",
        max_tokens=256,
    )
    if err1:
        result.errors.append(f"simple/thinking_off: {err1}")
    result.simple_ok = ok1
    result.thinking_off_ok = ok1
    result.simple_latency_s = lat1 if ok1 else None
    result.simple_answer = text1
    result.simple_chars = len(text1)
    json1_ok, obj1 = parse_jsonish(text1)

    ok2, lat2, text2, err2 = chat_once(
        base_url,
        api_key,
        model,
        COMPLEX_PROMPT,
        enable_thinking=True,
        verbosity="medium",
        max_tokens=700,
    )
    if err2:
        result.errors.append(f"complex/thinking_on: {err2}")
    result.complex_ok = ok2
    result.thinking_on_ok = ok2
    result.complex_latency_s = lat2 if ok2 else None
    result.complex_answer = text2
    result.complex_chars = len(text2)
    json2_ok, obj2 = parse_jsonish(text2)

    result.json_ok = json1_ok and json2_ok
    result.verbosity_ok = (
        json1_ok
        and json2_ok
        and len(text1) <= 320
        and isinstance(obj2, dict)
        and isinstance(obj2.get("risks"), list)
    )
    result.ok = result.simple_ok and result.complex_ok and result.json_ok
    score_result(result)
    return result


def top_three(results: List[TestResult], key: str) -> List[TestResult]:
    candidates = [
        r
        for r in results
        if r.ok
        and r.thinking_off_ok
        and r.thinking_on_ok
        and r.verbosity_ok
        and not r.excluded_qwen_deepseek
    ]
    return sorted(candidates, key=lambda r: getattr(r, key), reverse=True)[:3]


def write_results(
    path: str,
    base_url: str,
    all_models: List[str],
    chat_candidates: List[str],
    results: List[TestResult],
) -> None:
    out = {
        "tested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": base_url,
        "model_count": len(all_models),
        "candidate_count": len(chat_candidates),
        "filtered_out": [m for m in all_models if m not in chat_candidates],
        "top_simple": [asdict(r) for r in top_three(results, "score_simple")],
        "top_complex": [asdict(r) for r in top_three(results, "score_complex")],
        "all_results": [asdict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Only test the first N candidate models.")
    parser.add_argument("--models", nargs="*", help="Only test these exact model IDs.")
    parser.add_argument("--out", default="api_test_results.json")
    args = parser.parse_args()

    env = load_env()
    api_key = os.environ.get("SILICONFLOW_API_KEY") or env["SILICONFLOW_API_KEY"]
    base_url = (
        os.environ.get("SILICONFLOW_API_URL")
        or env.get("SILICONFLOW_API_URL")
        or "https://api.siliconflow.com/v1"
    )

    all_models = list_models(base_url, api_key)
    chat_candidates = [m for m in all_models if not EXCLUDE_KIND_RE.search(m)]
    if args.models:
        wanted = set(args.models)
        chat_candidates = [m for m in chat_candidates if m in wanted]
    if args.limit > 0:
        chat_candidates = chat_candidates[: args.limit]

    print(f"Fetched models: {len(all_models)}", flush=True)
    print(f"Candidate chat models after non-chat filter: {len(chat_candidates)}", flush=True)

    results: List[TestResult] = []
    for idx, model in enumerate(chat_candidates, 1):
        print(f"[{idx}/{len(chat_candidates)}] testing {model}", flush=True)
        result = test_model(base_url, api_key, model)
        results.append(result)
        status = "OK" if result.ok else "FAIL"
        print(
            f"  {status} simple={result.simple_latency_s} complex={result.complex_latency_s} "
            f"json={result.json_ok} score_simple={result.score_simple:.1f} "
            f"score_complex={result.score_complex:.1f}",
            flush=True,
        )
        write_results(args.out, base_url, all_models, chat_candidates, results)

    write_results(args.out, base_url, all_models, chat_candidates, results)

    print("\nTop simple:")
    for r in top_three(results, "score_simple"):
        print(f"- {r.model}: score={r.score_simple:.1f}, latency={r.simple_latency_s:.2f}s")

    print("\nTop complex:")
    for r in top_three(results, "score_complex"):
        print(f"- {r.model}: score={r.score_complex:.1f}, latency={r.complex_latency_s:.2f}s")


if __name__ == "__main__":
    main()
