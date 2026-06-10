from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from app.models.chunk import Chunk
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from app.models.chunk import Chunk


ITEM_LINE_PATTERN = re.compile(r"^\[(?P<type>[^\]]+)]\s+\[(?P<style>[^\]]+)]\s*(?P<text>.*)$")
DESCRIPTION_PATTERN = re.compile(r"^(?P<value>.*?)（(?P<description>.*)）$")
NUMBERING_PATTERN = re.compile(
    r"(?im)(?P<prefix>^|[\n\r]+|[ \t\u3000]+)"
    r"(?P<marker>"
    r"\d{1,3}\s*[.．、)]|"
    r"[一二三四五六七八九十百千万]{1,6}\s*[、.．)]|"
    r"第\s*[\d一二三四五六七八九十百千万]{1,6}\s*[章节篇部分]|"
    r"part\s*\d+|chapter\s*\d+|section\s*\d+"
    r")"
)
SENTENCE_END_PATTERN = re.compile(r"[。！？!?；;]\s*|[.]\s+(?=[^\d])|\n+")
MIN_CHUNK_CHARS = 10
MAX_CHUNK_CHARS = 500
FORCED_SPLIT_OVERLAP_CHARS = 20


@dataclass
class ChunkState:
    metadata: dict[str, Any] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)
    links: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    chunk_type: str = "text"


def split_txt_file(txt_file: str | Path) -> list[Chunk]:
    path = Path(txt_file)
    items = load_items_from_txt(path)
    return split_items(items, source_file=path)


def split_processing_txt(document_name: str) -> list[Chunk]:
    txt_file = Path("data") / "processing" / document_name / "txt" / f"{document_name}.txt"
    return split_txt_file(txt_file)


def save_chunks(chunks: list[Chunk], output_file: str | Path) -> Path:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [chunk.to_dict() for chunk in chunks]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_processing_chunks(document_name: str) -> Path:
    chunks = split_processing_txt(document_name)
    output_file = (
        Path("data")
        / "processing"
        / document_name
        / "chunk"
        / f"{document_name}.chunks.json"
    )
    return save_chunks(chunks, output_file)


def load_items_from_txt(txt_file: str | Path) -> list[dict[str, str]]:
    path = Path(txt_file)
    if not path.exists():
        raise FileNotFoundError(f"Parsed txt file not found: {path}")

    items: list[dict[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = ITEM_LINE_PATTERN.match(line)
        if match is None:
            items.append({"type": "paragraph", "style": "正文", "text": line})
            continue

        item = {
            "type": match.group("type"),
            "style": match.group("style"),
            "text": match.group("text").strip(),
        }
        _restore_special_item_fields(item)
        items.append(item)

    return items


def split_items(items: list[dict[str, str]], *, source_file: Path | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    state = ChunkState()
    heading_path: dict[int, str] = {}
    document_metadata = _build_document_metadata(source_file)

    for item_index, item in enumerate(items):
        item_type = item.get("type", "")
        style = item.get("style", "")
        text = item.get("text", "").strip()

        if not text:
            continue

        if _is_heading(style):
            _flush_chunk(chunks, state, document_metadata)
            _update_heading_path(heading_path, style, text)
            continue

        if not state.lines:
            state.metadata = _build_chunk_metadata(document_metadata, heading_path)

        if item_type == "image":
            _append_image(state, item, item_index)
        elif item_type == "link_ref":
            _append_link_ref(state, item, item_index)
        elif item_type == "link":
            _append_link(state, item, item_index)
        elif item_type in {"table", "image_table", "img_table"}:
            _append_table_chunk(chunks, state, item, heading_path, document_metadata, item_index)
        else:
            state.lines.append(text)

    _flush_chunk(chunks, state, document_metadata)
    return chunks


def _restore_special_item_fields(item: dict[str, str]) -> None:
    if item["type"] not in {"image", "link", "link_ref"}:
        return

    match = DESCRIPTION_PATTERN.match(item["text"])
    if match is None:
        value = item["text"]
        description = ""
    else:
        value = match.group("value").strip()
        description = match.group("description").strip()

    item["text"] = value
    if description:
        item["description"] = description

    if item["type"] == "image":
        item["path"] = value
    elif item["type"] in {"link", "link_ref"}:
        item["url"] = value


def _is_heading(style: str) -> bool:
    return style == "标题" or style.startswith("标题 ")


def _update_heading_path(heading_path: dict[int, str], style: str, text: str) -> None:
    level = _heading_level(style)
    heading_path[level] = text

    for old_level in list(heading_path):
        if old_level > level:
            heading_path.pop(old_level)


def _heading_level(style: str) -> int:
    match = re.search(r"\d+", style)
    return int(match.group(0)) if match else 1


def _append_image(state: ChunkState, item: dict[str, str], item_index: int) -> None:
    description = item.get("description") or item.get("text", "")
    if item.get("path"):
        state.images.append(_image_metadata(item["path"], item_index))
    if description.strip().lower() == "already satisfied":
        return
    state.lines.append(f"图片：{description}")


def _append_link(state: ChunkState, item: dict[str, str], item_index: int) -> None:
    description = item.get("description") or item.get("text", "")
    state.lines.append(f"链接：{description}")
    if item.get("url"):
        state.links.append(_link_metadata(description, item["url"], item_index))


def _append_link_ref(state: ChunkState, item: dict[str, str], item_index: int) -> None:
    description = item.get("description") or item.get("text", "")
    if description and item.get("url"):
        state.links.append(_link_metadata(description, item["url"], item_index))


def _append_table_chunk(
    chunks: list[Chunk],
    state: ChunkState,
    item: dict[str, Any],
    heading_path: dict[int, str],
    document_metadata: dict[str, Any],
    item_index: int,
) -> None:
    _flush_chunk(chunks, state, document_metadata)
    state.metadata = _build_chunk_metadata(document_metadata, heading_path)
    state.chunk_type = "table"
    if isinstance(item.get("links"), dict):
        for description, url in item["links"].items():
            if str(description).strip() and str(url).strip():
                state.links.append(_link_metadata(str(description), str(url), item_index))
    state.lines.append(str(item.get("text", "")).strip())
    _flush_chunk(chunks, state, document_metadata)


def _flush_chunk(chunks: list[Chunk], state: ChunkState, document_metadata: dict[str, Any]) -> None:
    content = "\n".join(line for line in state.lines if line).strip()
    if not content or _non_space_length(content) < MIN_CHUNK_CHARS:
        state.lines.clear()
        state.links.clear()
        state.images.clear()
        state.chunk_type = "text"
        return

    metadata = dict(state.metadata)
    if not metadata:
        metadata = _build_chunk_metadata(document_metadata, {})
    metadata["chunk_index"] = len(chunks)
    metadata["chunk_type"] = state.chunk_type
    if state.links:
        metadata["links"] = _dedupe_assets(state.links, "link_path")
    if state.images:
        metadata["imgs"] = _dedupe_assets(state.images, "img_path")

    for chunk_content in _split_content_for_limit(content, state.chunk_type):
        if _non_space_length(chunk_content) < MIN_CHUNK_CHARS:
            continue

        chunk_metadata = dict(metadata)
        chunk_metadata["chunk_index"] = len(chunks)
        chunks.append(
            Chunk(
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_index=len(chunks),
                file_id=str(chunk_metadata.get("file_id") or "") or None,
            )
        )

    state.lines.clear()
    state.links.clear()
    state.images.clear()
    state.metadata = {}
    state.chunk_type = "text"


def _non_space_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _split_content_for_limit(content: str, chunk_type: str) -> list[str]:
    if len(content) <= MAX_CHUNK_CHARS:
        return [content]

    if chunk_type == "table" or _looks_like_json(content):
        json_chunks = _split_json_content(content)
        if json_chunks:
            return json_chunks

    return _split_plain_text_content(content)


def _looks_like_json(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def _split_json_content(content: str) -> list[str]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return []

    chunks = _json_value_chunks(value)
    if not chunks:
        return []
    return [_json_text(chunk) for chunk in chunks]


def _json_value_chunks(value: Any) -> list[Any]:
    if len(_json_text(value)) <= MAX_CHUNK_CHARS:
        return [value]

    if isinstance(value, list):
        return _split_json_list(value)

    if isinstance(value, dict):
        list_key = _best_list_key(value)
        if list_key is not None:
            return _split_json_dict_list(value, list_key)
        return _split_json_dict_items(value)

    return [value]


def _split_json_list(rows: list[Any]) -> list[list[Any]]:
    chunks: list[list[Any]] = []
    current: list[Any] = []

    for row in rows:
        candidate = [*current, row]
        if current and len(_json_text(candidate)) > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = [row]
        else:
            current.append(row)

    if current:
        chunks.append(current)
    return chunks


def _best_list_key(value: dict[str, Any]) -> str | None:
    preferred_keys = ("table_data", "data", "rows", "items")
    for key in preferred_keys:
        if isinstance(value.get(key), list):
            return key
    for key, item in value.items():
        if isinstance(item, list):
            return key
    return None


def _split_json_dict_list(value: dict[str, Any], list_key: str) -> list[dict[str, Any]]:
    rows = value.get(list_key)
    if not isinstance(rows, list):
        return [value]

    chunks: list[dict[str, Any]] = []
    current: list[Any] = []

    for row in rows:
        candidate_rows = [*current, row]
        candidate = {**value, list_key: candidate_rows}
        if current and len(_json_text(candidate)) > MAX_CHUNK_CHARS:
            chunks.append({**value, list_key: current})
            current = [row]
        else:
            current.append(row)

    if current:
        chunks.append({**value, list_key: current})
    return chunks


def _split_json_dict_items(value: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: dict[str, Any] = {}

    for key, item in value.items():
        candidate = {**current, key: item}
        if current and len(_json_text(candidate)) > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = {key: item}
        else:
            current[key] = item

    if current:
        chunks.append(current)
    return chunks


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _split_plain_text_content(content: str) -> list[str]:
    chunks: list[str] = []
    remaining = content.strip()

    while len(remaining) > MAX_CHUNK_CHARS:
        split_at = _numbering_split_point(remaining, MAX_CHUNK_CHARS)
        if split_at is None:
            split_at = _sentence_end_split_point(remaining, MAX_CHUNK_CHARS)
        if split_at is not None:
            chunk = remaining[:split_at].rstrip()
            remaining = remaining[split_at:].lstrip()
        else:
            chunk = remaining[:MAX_CHUNK_CHARS].rstrip()
            next_start = max(0, MAX_CHUNK_CHARS - FORCED_SPLIT_OVERLAP_CHARS)
            remaining = remaining[next_start:].lstrip()

        if chunk:
            chunks.append(chunk)

    if remaining:
        chunks.append(remaining)
    return chunks


def _numbering_split_point(text: str, max_chars: int) -> int | None:
    split_at: int | None = None
    for match in NUMBERING_PATTERN.finditer(text[:max_chars]):
        marker_start = match.start("marker")
        if marker_start > MIN_CHUNK_CHARS:
            split_at = marker_start
    return split_at


def _sentence_end_split_point(text: str, max_chars: int) -> int | None:
    split_at: int | None = None
    for match in SENTENCE_END_PATTERN.finditer(text[:max_chars]):
        end = match.end()
        if end > MIN_CHUNK_CHARS:
            split_at = end
    return split_at


def _build_document_metadata(source_file: Path | None) -> dict[str, Any]:
    if source_file is None:
        file_path = ""
        file_name = ""
        file_type = "unknown"
        name_for_hash = "unknown"
    else:
        path = Path(source_file)
        file_path = str(path)
        file_name = path.name
        file_type = _infer_file_type(path)
        name_for_hash = path.stem or "unknown"

    return {
        "file_id": build_file_id(file_type, name_for_hash),
        "file_type": file_type,
        "file_name": file_name,
        "file_path": file_path,
    }


def build_file_id(file_type: str, file_stem: str) -> str:
    digest = hashlib.sha1(file_stem.encode("utf-8")).hexdigest()[:12]
    return f"{file_type}_{digest}"


def _infer_file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return "doc"
    return suffix.lstrip(".") or "unknown"


def _build_chunk_metadata(
    document_metadata: dict[str, Any],
    heading_path: dict[int, str],
) -> dict[str, Any]:
    metadata = dict(document_metadata)
    metadata["path"] = _format_heading_path(heading_path)
    return metadata


def _format_heading_path(heading_path: dict[int, str]) -> str:
    return "/".join(
        _safe_path_segment(text)
        for _, text in sorted(heading_path.items())
        if _safe_path_segment(text)
    )


def _safe_path_segment(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    return normalized.replace("/", "／").replace("\\", "＼")


def _image_metadata(path: str, item_index: int) -> dict[str, Any]:
    return {
        "index": item_index,
        "img_name": Path(path).name,
        "img_path": path,
    }


def _link_metadata(description: str, url: str, item_index: int) -> dict[str, Any]:
    return {
        "index": item_index,
        "link_name": description,
        "link_path": url,
    }


def _dedupe_assets(items: list[dict[str, Any]], path_key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        path = str(item.get(path_key) or "")
        name = str(item.get("img_name") or item.get("link_name") or "")
        key = f"{path}\n{name}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(item))
    return deduped


if __name__ == "__main__":
    target_txt = Path("data") / "processing" / "订阅号运营SOP" / "txt" / "订阅号运营SOP.txt"
    chunks = split_txt_file(target_txt)
    output_path = save_chunks(
        chunks,
        target_txt.parents[1] / "chunk" / f"{target_txt.stem}.chunks.json",
    )

    print(f"File: {target_txt}")
    print(f"Chunks: {len(chunks)}")
    print(f"Chunk output: {output_path}")
    for index, chunk in enumerate(chunks, start=1):
        print("-" * 80)
        print(f"Chunk {index}")
        print(chunk.metadata)
        print(chunk.content)
