from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

from app.core.config import settings
from app.db.minio import parse_raw_document_reference


_REFERENCE_TAG = re.compile(
    r"<reference\b[^>]*>(.*?)</reference>",
    re.IGNORECASE | re.DOTALL,
)
_PSEUDO_TAG = re.compile(r"<(img|link|a)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_SOURCE_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?(知识来源|引用文献|参考来源)(?:\*\*)?\s*[:：]?\s*$"
)
_LOCAL_TO_LARK_MAPPING_DIR = Path("data") / "metadata" / "local2lark_mapping"
_local_to_lark_mapping_cache: dict[str, str] | None = None


def format_external_markdown_answer(
    answer: str,
    *,
    use_lark_document: bool,
) -> str:
    """Convert agent pseudo-tags into portable Markdown with a source section."""

    markdown = _strip_model_generated_source_section(answer)
    sources: list[tuple[str, str]] = []
    source_numbers: dict[str, int] = {}
    in_code_block = False
    output_lines: list[str] = []

    def replace_reference(match: re.Match[str]) -> str:
        if in_code_block:
            return match.group(0)
        raw_path = match.group(1).strip()
        if not raw_path:
            return ""
        normalized = _normalize_reference(raw_path)
        if normalized not in source_numbers:
            source_numbers[normalized] = len(sources) + 1
            sources.append((raw_path, _external_reference_url(raw_path, use_lark_document)))
        return f"[{source_numbers[normalized]}]"

    for line in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            output_lines.append(line)
            continue
        rewritten = _REFERENCE_TAG.sub(replace_reference, line)
        if not in_code_block:
            rewritten = _PSEUDO_TAG.sub(
                lambda match: _replace_pseudo_tag(
                    match,
                    use_lark_document=use_lark_document,
                ),
                rewritten,
            )
        output_lines.append(rewritten)

    body = "\n".join(output_lines).strip()
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if not sources:
        return body

    source_lines = ["---", "### 知识来源"]
    for index, (raw_path, url) in enumerate(sources, start=1):
        description = _extract_file_name(raw_path) or f"来源 {index}"
        source_lines.append(f"{index}. [{_escape_markdown_label(description)}]({url})")
    return f"{body}\n\n" + "\n".join(source_lines)


def _replace_pseudo_tag(match: re.Match[str], *, use_lark_document: bool) -> str:
    tag = match.group(1).lower()
    raw_path = match.group(2).strip()
    if not raw_path:
        return ""
    url = _external_reference_url(raw_path, use_lark_document)
    label = _escape_markdown_label(_extract_file_name(raw_path) or "链接")
    if tag == "img":
        return f"![{label}]({url})"
    return f"[{label}]({url})"


def _strip_model_generated_source_section(markdown: str) -> str:
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    in_code_block = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if not in_code_block and _SOURCE_HEADING.match(stripped):
            cutoff = index
            while cutoff > 0 and not lines[cutoff - 1].strip():
                cutoff -= 1
            if cutoff > 0 and re.fullmatch(
                r"[-*_]\s*[-*_]\s*[-*_\s]*",
                lines[cutoff - 1].strip(),
            ):
                cutoff -= 1
            return "\n".join(lines[:cutoff]).rstrip()
    return markdown


def _external_reference_url(raw_path: str, use_lark_document: bool) -> str:
    normalized = _normalize_reference(raw_path)
    parsed = urlparse(normalized)
    if parsed.scheme in {"http", "https"}:
        return normalized
    if use_lark_document:
        lark_url = _resolve_lark_document_url(normalized)
        if lark_url:
            return lark_url
    query = urlencode(
        {"path": _normalize_download_reference(normalized)},
        quote_via=quote,
    )
    return f"{_public_api_base_url()}/api/v1/documents/download?{query}"


def _public_api_base_url() -> str:
    base_url = (settings.public_base_url or "http://localhost:8000").rstrip("/")
    prefix = settings.api_route_prefix.strip()
    if not prefix:
        return base_url
    prefix = f"/{prefix.strip('/')}"
    if urlparse(base_url).path.rstrip("/").endswith(prefix):
        return base_url
    return f"{base_url}{prefix}"


def _normalize_download_reference(raw_path: str) -> str:
    parsed = urlparse(raw_path)
    if parsed.query:
        query_path = (parse_qs(parsed.query).get("path") or [""])[0]
        if query_path:
            raw_path = query_path
    try:
        return parse_raw_document_reference(raw_path).uri
    except ValueError:
        return raw_path


def _resolve_lark_document_url(raw_path: str) -> str | None:
    file_name = _extract_file_name(raw_path)
    if not file_name:
        return None
    return _load_local_to_lark_mapping().get(_normalize_document_name(file_name))


def _load_local_to_lark_mapping() -> dict[str, str]:
    global _local_to_lark_mapping_cache
    if _local_to_lark_mapping_cache is not None:
        return _local_to_lark_mapping_cache

    mapping: dict[str, str] = {}
    if _LOCAL_TO_LARK_MAPPING_DIR.exists():
        for mapping_path in sorted(_LOCAL_TO_LARK_MAPPING_DIR.glob("*.json")):
            try:
                data = json.loads(mapping_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            for file_name, lark_url in data.items():
                if isinstance(file_name, str) and isinstance(lark_url, str):
                    mapping.setdefault(_normalize_document_name(file_name), lark_url)
    _local_to_lark_mapping_cache = mapping
    return mapping


def _extract_file_name(raw_path: str) -> str:
    normalized = unquote(_normalize_reference(raw_path)).strip().strip("\"'")
    parsed = urlparse(normalized)
    query_path = (parse_qs(parsed.query).get("path") or [""])[0] if parsed.query else ""
    candidate = unquote(query_path or parsed.path or normalized).replace("\\", "/")
    return PurePosixPath(candidate.strip().strip("\"'")).name


def _normalize_reference(value: str) -> str:
    return value.replace("\\", "/").strip()


def _normalize_document_name(value: str) -> str:
    return re.sub(r"\s+", "", value).strip().lower()


def _escape_markdown_label(value: str) -> str:
    return value.replace("[", r"\[").replace("]", r"\]")
