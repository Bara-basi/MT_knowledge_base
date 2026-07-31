from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from app.db.minio import parse_raw_document_reference
from app.models.chunk import Chunk
from app.services.data_clean import clean_items


ITEM_LINE_PATTERN = re.compile(r"^\[(?P<type>[^\]]+)]\s+\[(?P<style>[^\]]+)]\s*(?P<text>.*)$")
DESCRIPTION_PATTERN = re.compile(r"^(?P<value>.*?)（(?P<description>.*)）$")
IMG_TAG_PATTERN = re.compile(r'(?s)<img\s+index="(?P<index>\d+)">(?P<body>.*?)</img>')
LINK_TAG_PATTERN = re.compile(r'(?s)<a\s+index="(?P<index>\d+)">.*?</a>')
ASSET_TAG_PATTERN = re.compile(
    r'(?s)<(?P<tag>img|a)\s+index="(?P<index>\d+)">(?P<body>.*?)</(?P=tag)>'
)
LINK_MARKER_PATTERN = re.compile(r"\{\{[^{}\r\n]*\}\}")
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
SHORT_CHUNK_CHARS = 60
MAX_CHUNK_CHARS = 500
FORCED_SPLIT_OVERLAP_CHARS = 20
PATH_SEPARATOR = "\\"


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


def split_items(items: list[dict[str, str]], *, source_file: str | Path | None = None) -> list[Chunk]:
    items = clean_items(items)
    chunks: list[Chunk] = []
    state = ChunkState()
    heading_path: dict[int, str] = {}
    pending_table_title = ""
    previous_heading_level: int | None = None
    document_metadata = _build_document_metadata(source_file)

    for item_index, item in enumerate(items):
        item_type = item.get("type", "")
        style = item.get("style", "")
        text = item.get("text", "").strip()

        if not text:
            continue

        if _is_table_title(item_type, style):
            _flush_chunk(chunks, state, document_metadata)
            pending_table_title = text
            previous_heading_level = None
            continue

        if _is_heading(style):
            _flush_chunk(chunks, state, document_metadata)
            level = _heading_level(style)
            _update_heading_path(
                heading_path,
                style,
                text,
                append_same_level=(previous_heading_level == level),
            )
            previous_heading_level = level
            pending_table_title = ""
            continue

        previous_heading_level = None
        if pending_table_title and item_type not in {"table", "image_table", "img_table"}:
            pending_table_title = ""

        if not state.lines:
            state.metadata = _build_chunk_metadata(document_metadata, heading_path)

        if item_type == "image":
            _append_image(state, item, item_index)
        elif item_type == "link_ref":
            _append_link_ref(state, item, item_index)
        elif item_type == "link":
            _append_link(state, item, item_index)
        elif item_type in {"table", "image_table", "img_table"}:
            _append_table_chunk(chunks, state, item, heading_path, document_metadata, item_index, pending_table_title)
            pending_table_title = ""
        else:
            state.lines.append(_strip_link_markers(text))

    _flush_chunk(chunks, state, document_metadata)
    return _merge_short_text_chunks(chunks)


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


def _is_table_title(item_type: str, style: str) -> bool:
    return item_type == "table" and style == "表标题"


def _update_heading_path(
    heading_path: dict[int, str],
    style: str,
    text: str,
    *,
    append_same_level: bool = False,
) -> None:
    level = _heading_level(style)
    previous = heading_path.get(level, "")
    if append_same_level and previous:
        heading_path[level] = _join_heading_fragments(previous, text)
    else:
        heading_path[level] = text

    for old_level in list(heading_path):
        if old_level > level:
            heading_path.pop(old_level)


def _join_heading_fragments(previous: str, current: str) -> str:
    left = previous.strip()
    right = current.strip()
    if not left:
        return right
    if not right or right == left or right in left:
        return left
    if left in right:
        return right
    return f"{left} {right}"


def _heading_level(style: str) -> int:
    match = re.search(r"\d+", style)
    return int(match.group(0)) if match else 1


def _append_image(state: ChunkState, item: dict[str, str], item_index: int) -> None:
    description = item.get("description") or item.get("text", "")
    if item.get("path"):
        state.images.append(_image_metadata(item["path"], item_index))
    if description.strip().lower() == "already satisfied":
        return
    state.lines.append(_format_image_description_tag(item_index, description))


def _append_link(state: ChunkState, item: dict[str, str], item_index: int) -> None:
    description = item.get("description") or item.get("text", "")
    if item.get("url"):
        state.links.append(_link_metadata(description, item["url"], item_index))
        state.lines.append(_format_link_tag(item_index, description))


def _append_link_ref(state: ChunkState, item: dict[str, str], item_index: int) -> None:
    description = item.get("description") or item.get("text", "")
    if description and item.get("url"):
        state.links.append(_link_metadata(description, item["url"], item_index))
        state.lines.append(_format_link_tag(item_index, description))


def _append_table_chunk(
    chunks: list[Chunk],
    state: ChunkState,
    item: dict[str, Any],
    heading_path: dict[int, str],
    document_metadata: dict[str, Any],
    item_index: int,
    table_title: str = "",
) -> None:
    _flush_chunk(chunks, state, document_metadata)
    state.metadata = _build_chunk_metadata(document_metadata, heading_path)
    if table_title:
        state.metadata["path"] = _append_path_segment(state.metadata.get("path", ""), table_title)
    state.chunk_type = "table"
    table_link_indexes: list[tuple[int, str]] = []
    if isinstance(item.get("links"), dict):
        for offset, (description, url) in enumerate(item["links"].items(), start=1):
            if str(description).strip() and str(url).strip():
                link_index = _table_link_index(item_index, offset)
                state.links.append(_link_metadata(str(description), str(url), link_index))
                table_link_indexes.append((link_index, str(description)))
    state.lines.append(str(item.get("text", "")).strip())
    for link_index, description in table_link_indexes:
        state.lines.append(_format_link_tag(link_index, description))
    _flush_chunk(chunks, state, document_metadata)


def _table_link_index(item_index: int, offset: int) -> int:
    return item_index * 10000 + offset


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

        chunk_metadata = _metadata_for_chunk_content(metadata, chunk_content)
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


def _merge_short_text_chunks(chunks: list[Chunk]) -> list[Chunk]:
    merged_chunks = _reindex_chunks(chunks)
    for _ in range(8):
        next_chunks = _merge_short_text_chunks_once(merged_chunks)
        if _chunk_signature(next_chunks) == _chunk_signature(merged_chunks):
            return next_chunks
        merged_chunks = next_chunks
    return merged_chunks


def _merge_short_text_chunks_once(chunks: list[Chunk]) -> list[Chunk]:
    folded_paths = _short_text_paths_to_fold(chunks)
    if not folded_paths:
        return _reindex_chunks(chunks)

    merged_chunks: list[Chunk] = []
    run: list[Chunk] = []
    run_path = ""

    for chunk in chunks:
        prepared = _prepare_chunk_for_short_merge(chunk, folded_paths)
        if _can_merge_text_chunk(prepared):
            path = str(prepared.metadata.get("path") or "")
            if run and path != run_path:
                _flush_text_merge_run(merged_chunks, run)
                run = []
            run.append(prepared)
            run_path = path
            continue

        _flush_text_merge_run(merged_chunks, run)
        run = []
        run_path = ""
        merged_chunks.append(prepared)

    _flush_text_merge_run(merged_chunks, run)
    return _reindex_chunks(merged_chunks)


def _chunk_signature(chunks: list[Chunk]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            chunk.content,
            str(chunk.metadata.get("path") or ""),
            str(chunk.metadata.get("chunk_type") or ""),
        )
        for chunk in chunks
    )


def _short_text_paths_to_fold(chunks: list[Chunk]) -> dict[str, str]:
    folded: dict[str, str] = {}
    for chunk in chunks:
        if not _can_merge_text_chunk(chunk):
            continue
        path = str(chunk.metadata.get("path") or "")
        if _non_space_length(chunk.content) >= SHORT_CHUNK_CHARS:
            continue
        parent_path, _ = _split_path_leaf(path)
        if parent_path:
            folded[path] = parent_path
    return folded


def _prepare_chunk_for_short_merge(chunk: Chunk, folded_paths: dict[str, str]) -> Chunk:
    path = str(chunk.metadata.get("path") or "")
    new_path = _rewrite_folded_path(path, folded_paths)
    metadata = dict(chunk.metadata)
    if new_path != path:
        metadata["path"] = new_path

    if not _can_merge_text_chunk(chunk):
        return Chunk(
            content=chunk.content,
            metadata=metadata,
            chunk_index=chunk.chunk_index,
            file_id=chunk.file_id,
            vector_id=chunk.vector_id,
        )

    if path in folded_paths:
        _, leaf = _split_path_leaf(path)
        content = f"{leaf}\n{chunk.content}" if leaf else chunk.content
    else:
        content = chunk.content

    return Chunk(
        content=content,
        metadata=metadata,
        chunk_index=chunk.chunk_index,
        file_id=chunk.file_id,
        vector_id=chunk.vector_id,
    )


def _rewrite_folded_path(path: str, folded_paths: dict[str, str]) -> str:
    if not path:
        return path

    for old_path in sorted(folded_paths, key=len, reverse=True):
        new_path = folded_paths[old_path]
        if path == old_path:
            return new_path
        prefix = f"{old_path}{PATH_SEPARATOR}"
        if path.startswith(prefix):
            return _join_existing_path(new_path, path[len(prefix) :])
    return path


def _can_merge_text_chunk(chunk: Chunk) -> bool:
    chunk_type = str(chunk.metadata.get("chunk_type") or "text")
    return chunk_type == "text" and not _looks_like_json(chunk.content)


def _flush_text_merge_run(chunks: list[Chunk], run: list[Chunk]) -> None:
    if not run:
        return

    metadata = _merge_text_run_metadata(run)
    content = "\n".join(chunk.content.strip() for chunk in run if chunk.content.strip()).strip()
    for chunk_content in _split_content_for_limit(content, "text"):
        if _non_space_length(chunk_content) < MIN_CHUNK_CHARS:
            continue
        chunk_metadata = _metadata_for_chunk_content(metadata, chunk_content)
        chunks.append(
            Chunk(
                content=chunk_content,
                metadata=chunk_metadata,
                file_id=str(chunk_metadata.get("file_id") or "") or None,
            )
        )


def _merge_text_run_metadata(run: list[Chunk]) -> dict[str, Any]:
    metadata = dict(run[0].metadata)
    links: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    for chunk in run:
        links.extend(_metadata_list(chunk.metadata.get("links")))
        images.extend(_metadata_list(chunk.metadata.get("imgs")))
    if links:
        metadata["links"] = _dedupe_assets(links, "link_path")
    else:
        metadata.pop("links", None)
    if images:
        metadata["imgs"] = _dedupe_assets(images, "img_path")
    else:
        metadata.pop("imgs", None)
    metadata["chunk_type"] = "text"
    return metadata


def _metadata_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _reindex_chunks(chunks: list[Chunk]) -> list[Chunk]:
    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index
        chunk.metadata["chunk_index"] = index
        chunk.file_id = str(chunk.metadata.get("file_id") or "") or None
    return chunks


def _non_space_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _split_content_for_limit(content: str, chunk_type: str) -> list[str]:
    if len(content) <= MAX_CHUNK_CHARS:
        return [content]

    if chunk_type == "text" and _has_asset_tag(content):
        return _split_tagged_text_content(content)

    if chunk_type == "table" or _looks_like_json(content):
        json_chunks = _split_json_content(content)
        if json_chunks:
            return json_chunks

    return _split_plain_text_content(content)


def _has_asset_tag(text: str) -> bool:
    return ASSET_TAG_PATTERN.search(text) is not None


def _split_tagged_text_content(content: str) -> list[str]:
    chunks: list[str] = []
    cursor = 0

    for match in ASSET_TAG_PATTERN.finditer(content):
        before = content[cursor : match.start()].strip()
        if before:
            chunks.extend(_split_plain_text_content(before))
        chunks.extend(
            _split_asset_tag_block(
                match.group(0),
                match.group("tag"),
                match.group("index"),
                match.group("body"),
            )
        )
        cursor = match.end()

    after = content[cursor:].strip()
    if after:
        chunks.extend(_split_plain_text_content(after))
    return [chunk for chunk in chunks if chunk.strip()]


def _split_asset_tag_block(block: str, tag: str, asset_index: str, body: str) -> list[str]:
    if len(block) <= MAX_CHUNK_CHARS:
        return [block.strip()]

    if tag != "img":
        return [block.strip()]

    open_tag = f'<img index="{asset_index}">'
    close_tag = "</img>"
    max_body_chars = MAX_CHUNK_CHARS - len(open_tag) - len(close_tag)
    if max_body_chars <= MIN_CHUNK_CHARS:
        return [block.strip()]

    return [
        f"{open_tag}{part.strip()}{close_tag}"
        for part in _split_plain_text_content_with_limit(body.strip(), max_body_chars)
        if part.strip()
    ]


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


def _split_json_dict_items(value: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: dict[str, Any] = {}

    for key, item in value.items():
        single_item = {key: item}
        if len(_json_text(single_item)) > MAX_CHUNK_CHARS:
            if current:
                chunks.append(current)
                current = {}
            chunks.extend(_split_json_dict_item(key, item))
            continue

        candidate = {**current, key: item}
        if current and len(_json_text(candidate)) > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = {key: item}
        else:
            current[key] = item

    if current:
        chunks.append(current)
    return chunks


def _split_json_dict_item(key: str, item: Any) -> list[dict[str, Any]]:
    if isinstance(item, list):
        return [{key: chunk} for chunk in _split_wrapped_json_list(key, item)]

    if isinstance(item, dict):
        return [{key: chunk} for chunk in _split_json_dict_items(item)]

    return [{key: item}]


def _split_wrapped_json_list(key: str, rows: list[Any]) -> list[list[Any]]:
    chunks: list[list[Any]] = []
    current: list[Any] = []

    for row in rows:
        candidate = [*current, row]
        if current and len(_json_text({key: candidate})) > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = [row]
        else:
            current.append(row)

    if current:
        chunks.append(current)
    return chunks


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _split_plain_text_content(content: str) -> list[str]:
    return _split_plain_text_content_with_limit(content, MAX_CHUNK_CHARS)


def _split_plain_text_content_with_limit(content: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    remaining = content.strip()

    while len(remaining) > max_chars:
        split_at = _numbering_split_point(remaining, max_chars)
        if split_at is None:
            split_at = _sentence_end_split_point(remaining, max_chars)
        if split_at is not None:
            chunk = remaining[:split_at].rstrip()
            remaining = remaining[split_at:].lstrip()
        else:
            chunk = remaining[:max_chars].rstrip()
            next_start = max(0, max_chars - FORCED_SPLIT_OVERLAP_CHARS)
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


def _build_document_metadata(source_file: str | Path | None) -> dict[str, Any]:
    if source_file is None:
        file_path = ""
        file_name = ""
        file_type = "unknown"
        name_for_hash = "unknown"
    elif _looks_like_minio_document_source(str(source_file)):
        reference = parse_raw_document_reference(str(source_file))
        object_path = PurePosixPath(reference.object_name)
        file_path = reference.uri
        file_name = object_path.name
        file_type = _infer_file_type_from_suffix(object_path.suffix)
        name_for_hash = object_path.stem or "unknown"
    else:
        path = Path(source_file)
        file_path = str(path)
        file_name = path.name
        file_type = _infer_file_type_from_suffix(path.suffix)
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


def _infer_file_type_from_suffix(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix == ".docx":
        return "doc"
    return suffix.lstrip(".") or "unknown"


def _looks_like_minio_document_source(value: str) -> bool:
    normalized = value.strip().replace("\\", "/")
    parsed = urlparse(normalized)
    return parsed.scheme in {"minio", "s3"} or "data/raw/" in normalized.lower()


def _build_chunk_metadata(
    document_metadata: dict[str, Any],
    heading_path: dict[int, str],
) -> dict[str, Any]:
    metadata = dict(document_metadata)
    metadata["path"] = _format_heading_path(heading_path)
    return metadata


def _format_heading_path(heading_path: dict[int, str]) -> str:
    return PATH_SEPARATOR.join(
        _safe_path_segment(text)
        for _, text in sorted(heading_path.items())
        if _safe_path_segment(text)
    )


def _safe_path_segment(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    return normalized.replace("/", "／").replace("\\", "＼")


def _append_path_segment(path: str, segment: str) -> str:
    safe_segment = _safe_path_segment(segment)
    if not safe_segment:
        return path
    return f"{path}{PATH_SEPARATOR}{safe_segment}" if path else safe_segment


def _join_existing_path(path: str, suffix: str) -> str:
    suffix = str(suffix or "").strip(PATH_SEPARATOR)
    if not suffix:
        return path
    return f"{path}{PATH_SEPARATOR}{suffix}" if path else suffix


def _split_path_leaf(path: str) -> tuple[str, str]:
    normalized = str(path or "").strip(PATH_SEPARATOR)
    if not normalized:
        return "", ""
    if PATH_SEPARATOR not in normalized:
        return "", normalized
    parent, leaf = normalized.rsplit(PATH_SEPARATOR, 1)
    return parent, leaf


def _format_image_description_tag(item_index: int, description: str) -> str:
    safe_description = str(description).replace("</img>", "<／img>")
    return f'<img index="{item_index}">图片：{safe_description}</img>'


def _format_link_tag(item_index: int, description: str) -> str:
    safe_description = str(description).replace("</a>", "<／a>")
    return f'<a index="{item_index}">链接：{safe_description}</a>'


def _strip_link_markers(text: str) -> str:
    return LINK_MARKER_PATTERN.sub("", text).strip()


def _metadata_for_chunk_content(metadata: dict[str, Any], content: str) -> dict[str, Any]:
    chunk_metadata = dict(metadata)
    if "links" in chunk_metadata:
        link_indexes = _link_indexes_in_content(content)
        links = [
            dict(item)
            for item in _metadata_list(chunk_metadata.get("links"))
            if _metadata_item_index(item) in link_indexes
        ]
        if links:
            chunk_metadata["links"] = _dedupe_assets(links, "link_path")
        else:
            chunk_metadata.pop("links", None)

    if "imgs" not in chunk_metadata:
        return chunk_metadata

    image_indexes = _image_indexes_in_content(content)
    imgs = [
        dict(item)
        for item in _metadata_list(chunk_metadata.get("imgs"))
        if _metadata_item_index(item) in image_indexes
    ]
    if imgs:
        chunk_metadata["imgs"] = _dedupe_assets(imgs, "img_path")
    else:
        chunk_metadata.pop("imgs", None)
    return chunk_metadata


def _image_indexes_in_content(content: str) -> set[int]:
    indexes: set[int] = set()
    for match in IMG_TAG_PATTERN.finditer(content):
        indexes.add(int(match.group("index")))
    return indexes


def _link_indexes_in_content(content: str) -> set[int]:
    indexes: set[int] = set()
    for match in LINK_TAG_PATTERN.finditer(content):
        indexes.add(int(match.group("index")))
    return indexes


def _metadata_item_index(item: dict[str, Any]) -> int:
    try:
        return int(item.get("index", -1))
    except (TypeError, ValueError):
        return -1


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
