"""Extract high-confidence, exact alias pairs from the curated source manifest.

The manifest deliberately excludes the deferred batches (generic standards, HR,
administration, and broad industry research).  Results are candidates for human
review; no query-rewrite rule is published directly by this script.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe scripts\\alias\\extract_aliases_kimi.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data/metadata/同名词库/alias_extraction_sources.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/metadata/同名词库"

SYSTEM_PROMPT = """你是工业金属材料企业知识库的术语归一专家。你的任务是找出同一实体的\"精确别名\"，供关键词检索时互相替换。

只保留以下关系：中英文对照、正式名与公认简称、标准号的常见写法、UNS号/牌号/商业牌号、同一公司或工厂的中英文名、空格或连字符等书写变体。
严禁输出上下位词、相关词、产品组合、用途、不同制造方式或不同产品；例如 pipe 与 tube、无缝与焊接、法兰与管件均不能默认当作别名。若不能确定两个词可互换，请不要输出。

仅输出 JSON，不要 Markdown：
{"pairs":[{"original_term":"规范或较完整的名称","alias_term":"可替换名称","entity_type":"product|material_grade|standard|process|inspection|commercial_term|company|other","confidence":"high|medium","evidence":"不超过30字的原文证据"}]}
"""


@dataclass(frozen=True)
class Source:
    source_id: str
    category: str
    priority: str
    path: Path
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=None, help="Defaults to KIMI_MODEL or kimi-k2.6")
    parser.add_argument("--max-chars-per-source", type=int, default=24000)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None, help="Useful for a small trial run")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[Source]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sources = []
    for row in rows:
        source_path = REPO_ROOT / row["source_path"]
        sources.append(Source(row["source_id"], row["category"], row["priority"], source_path, row["description"]))
    return sources


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
    return path.read_text(encoding="utf-8", errors="ignore")


def representative_text(text: str, limit: int) -> str:
    """Keep headings and evenly sampled segments for long converted documents."""
    text = re.sub(r"\[(?:paragraph|table|link_ref)\] \[[^\]]+\] ?", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= limit:
        return text
    # Three evenly-spaced portions preserve introductions, tables, and later appendices.
    part = limit // 3
    middle_start = max(0, len(text) // 2 - part // 2)
    return "\n\n[中间内容节选]\n\n".join((text[:part], text[middle_start:middle_start + part], text[-part:]))


def parse_json_response(content: str) -> list[dict[str, Any]]:
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{\s*\"pairs\".*\}\s*$", content, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    pairs = data.get("pairs", [])
    return pairs if isinstance(pairs, list) else []


def normalized(value: str) -> str:
    return re.sub(r"[\s\-_/（）()\[\]【】,.，。:：]", "", value).casefold()


def valid_pair(pair: dict[str, Any]) -> dict[str, str] | None:
    original = str(pair.get("original_term", "")).strip()
    alias = str(pair.get("alias_term", "")).strip()
    if not original or not alias or normalized(original) == normalized(alias):
        return None
    if len(original) > 100 or len(alias) > 100:
        return None
    return {
        "original_term": original,
        "alias_term": alias,
        "entity_type": str(pair.get("entity_type", "other")).strip() or "other",
        "confidence": str(pair.get("confidence", "medium")).strip() or "medium",
        "evidence": str(pair.get("evidence", "")).strip()[:120],
    }


def write_outputs(output_dir: Path, accepted: list[dict[str, str]], raw: list[dict[str, Any]], skipped: list[dict[str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alias_extraction_llm_responses.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in raw) + ("\n" if raw else ""), encoding="utf-8"
    )
    columns = ["original_term", "alias_term", "entity_type", "confidence", "sources", "evidence"]
    with (output_dir / "alias_candidates_kimi.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(accepted)
    (output_dir / "alias_extraction_skipped_sources.csv").write_text(
        "source_id,reason\n" + "\n".join(f'{x["source_id"]},{x["reason"]}' for x in skipped) + ("\n" if skipped else ""),
        encoding="utf-8-sig",
    )


def merge_pairs(merged: dict[tuple[str, str], dict[str, Any]], pairs: list[dict[str, Any]], source_id: str) -> None:
    for pair in pairs:
        clean = valid_pair(pair)
        if not clean:
            continue
        key = tuple(sorted((normalized(clean["original_term"]), normalized(clean["alias_term"]))))
        row = merged.setdefault(key, {**clean, "sources": set()})
        row["sources"].add(source_id)


def accepted_rows(merged: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, str]]:
    result = []
    for row in merged.values():
        result.append({**row, "sources": ";".join(sorted(row["sources"]))})
    return sorted(result, key=lambda row: (row["entity_type"], row["original_term"].casefold(), row["alias_term"].casefold()))


def load_checkpoint(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "alias_extraction_llm_responses.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict) and item.get("source_id") and isinstance(item.get("response"), str):
                records.append(item)
        except json.JSONDecodeError:
            continue
    return records


def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    sources = read_manifest(args.manifest)
    if args.limit:
        sources = sources[:args.limit]

    if args.dry_run:
        for source in sources:
            print(f"{source.priority}\t{source.source_id}\t{source.path}\t{'OK' if source.path.exists() else 'MISSING'}")
        return 0

    sys.path.insert(0, str(REPO_ROOT))
    from app.services.llm import LLMClient, LLMSettings  # Imported after dotenv loading.

    settings = LLMSettings.from_env()
    client = LLMClient(settings)
    model = args.model or settings.model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gathered = load_checkpoint(args.output_dir)
    completed = {item["source_id"] for item in gathered}
    skipped: list[dict[str, str]] = []
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in gathered:
        try:
            merge_pairs(merged, parse_json_response(item["response"]), item["source_id"])
        except (json.JSONDecodeError, TypeError):
            continue

    for index, source in enumerate(sources, start=1):
        if source.source_id in completed:
            print(f"[{index}/{len(sources)}] resume skip {source.source_id}")
            continue
        if not source.path.exists():
            skipped.append({"source_id": source.source_id, "reason": f"missing: {source.path.relative_to(REPO_ROOT)}"})
            print(f"[{index}/{len(sources)}] skip missing {source.source_id}")
            continue
        text = representative_text(read_text(source.path), args.max_chars_per_source)
        prompt = f"""来源编号：{source.source_id}
资料类别：{source.category}
资料用途：{source.description}

请从以下资料中抽取高置信度的精确别名对。资料可能是节选；没有别名时返回 {{\"pairs\":[]}}。

--- 资料开始 ---
{text}
--- 资料结束 ---"""
        print(f"[{index}/{len(sources)}] extracting {source.source_id} ({len(text)} chars)")
        try:
            response = client.chat(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                model=model,
                max_tokens=5000,
                # kimi-k2.6 otherwise may spend the entire completion budget in
                # reasoning_content and return an empty content field.
                extra_body={
                    "response_format": {"type": "json_object"},
                    "thinking": {"type": "disabled"},
                },
            )
            parsed = parse_json_response(response)
        except Exception as exc:  # Preserve other successful documents.
            skipped.append({"source_id": source.source_id, "reason": f"api_or_parse_error: {exc}"})
            print(f"  failed: {exc}")
            continue
        gathered.append({"source_id": source.source_id, "source_path": str(source.path.relative_to(REPO_ROOT)), "response": response, "pair_count": len(parsed)})
        merge_pairs(merged, parsed, source.source_id)
        # A successful source is immediately checkpointed, so another network
        # interruption resumes from the next source without losing prior work.
        write_outputs(args.output_dir, accepted_rows(merged), gathered, skipped)
        time.sleep(args.sleep_seconds)

    accepted = accepted_rows(merged)
    write_outputs(args.output_dir, accepted, gathered, skipped)
    print(f"Done: {len(accepted)} unique alias candidates from {len(gathered)} sources. Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
