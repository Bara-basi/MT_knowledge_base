from __future__ import annotations

import json
import re
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
STEP_PATTERN = re.compile(
    r"^(?P<marker>(?:第[一二三四五六七八九十百]+步)|(?:[一二三四五六七八九十]+、)|(?:\d+[、.．)])|(?:[①②③④⑤⑥⑦⑧⑨⑩]))"
)
MIN_CHUNK_CHARS = 10


@dataclass
class ChunkState:
    metadata: dict[str, Any] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)
    links: dict[str, str] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)


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
    structure: dict[str, str] = {}
    is_process_guide = _is_process_guide_source(source_file)

    for item in items:
        item_type = item.get("type", "")
        style = item.get("style", "")
        text = item.get("text", "").strip()

        if not text:
            continue

        if _is_heading(style):
            _flush_chunk(chunks, state, source_file)
            _update_heading_structure(structure, style, text, is_process_guide=is_process_guide)
            continue

        step = _extract_step(text) if is_process_guide else None
        if step:
            _flush_chunk(chunks, state, source_file)
            structure["step"] = step

        if not state.lines:
            state.metadata = dict(structure)

        if item_type == "image":
            _append_image(state, item)
        elif item_type == "link_ref":
            _append_link_ref(state, item)
        elif item_type == "link":
            _append_link(state, item)
        elif item_type in {"table", "image_table"}:
            _append_table_chunk(chunks, state, item, structure, source_file)
        else:
            state.lines.append(text)

    _flush_chunk(chunks, state, source_file)
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


def _is_process_guide_source(source_file: Path | None) -> bool:
    return source_file is not None and Path(source_file).parent.name == "process_guide"


def _update_heading_structure(
    structure: dict[str, str],
    style: str,
    text: str,
    *,
    is_process_guide: bool,
) -> None:
    if style == "标题":
        structure.clear()
        structure["title"] = text
        return

    level = _heading_level(style)
    key = "chapter" if is_process_guide and level <= 2 else f"heading_{level}"
    structure[key] = text

    for old_key in list(structure):
        if old_key.startswith("heading_") and int(old_key.split("_", 1)[1]) > level:
            structure.pop(old_key)
    structure.pop("step", None)


def _heading_level(style: str) -> int:
    match = re.search(r"\d+", style)
    return int(match.group(0)) if match else 1


def _extract_step(text: str) -> str | None:
    match = STEP_PATTERN.match(text)
    if not match:
        return None
    return match.group("marker")


def _append_image(state: ChunkState, item: dict[str, str]) -> None:
    description = item.get("description") or item.get("text", "")
    if item.get("path"):
        state.images.append(item["path"])
    if description.strip().lower() == "already satisfied":
        return
    state.lines.append(f"图片：{description}")


def _append_link(state: ChunkState, item: dict[str, str]) -> None:
    description = item.get("description") or item.get("text", "")
    state.lines.append(f"链接：{description}")
    if item.get("url"):
        state.links[description] = item["url"]


def _append_link_ref(state: ChunkState, item: dict[str, str]) -> None:
    description = item.get("description") or item.get("text", "")
    if description and item.get("url"):
        state.links[description] = item["url"]


def _append_table_chunk(
    chunks: list[Chunk],
    state: ChunkState,
    item: dict[str, Any],
    structure: dict[str, str],
    source_file: Path | None,
) -> None:
    _flush_chunk(chunks, state, source_file)
    state.metadata = dict(structure)
    if isinstance(item.get("links"), dict):
        state.links.update(
            {
                str(description): str(url)
                for description, url in item["links"].items()
                if str(description).strip() and str(url).strip()
            }
        )
    state.lines.append(str(item.get("text", "")).strip())
    _flush_chunk(chunks, state, source_file)


def _flush_chunk(chunks: list[Chunk], state: ChunkState, source_file: Path | None) -> None:
    content = "\n".join(line for line in state.lines if line).strip()
    if not content or _non_space_length(content) < MIN_CHUNK_CHARS:
        state.lines.clear()
        state.links.clear()
        state.images.clear()
        return

    metadata = dict(state.metadata)
    if source_file is not None:
        metadata["source_file"] = str(source_file)
    if state.links:
        metadata["link"] = dict(state.links)
    if state.images:
        metadata["img"] = list(dict.fromkeys(state.images))

    chunks.append(Chunk(content=content, metadata=metadata, chunk_index=len(chunks)))

    state.lines.clear()
    state.links.clear()
    state.images.clear()
    state.metadata = {}


def _non_space_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


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
