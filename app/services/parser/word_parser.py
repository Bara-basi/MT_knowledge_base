from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from docx import Document
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from docx.text.run import Run

try:
    from app.services.parser.img_parser import enrich_image_descriptions
    from app.services.parser.link_parser import enrich_links
except ModuleNotFoundError:
    from img_parser import enrich_image_descriptions
    from link_parser import enrich_links


IMAGE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/x-emf": ".emf",
    "image/x-wmf": ".wmf",
}
VML_IMAGE_DATA_TAG = "{urn:schemas-microsoft-com:vml}imagedata"
TABLE_JSON_MAX_CHARS = 800
EMPTY_CELL_TEXT = "（空）"
MAX_REPEATED_MERGED_CELL_CHARS = 15
PPT_LINK_PATTERN = re.compile(r"\[[^\[\]\r\n]*\.pptx?\]", re.IGNORECASE)
URL_TEXT_PATTERN = re.compile(r"^https?://\S+$", re.IGNORECASE)
LINK_STYLE = "链接"


@dataclass
class ParseContext:
    image_dir: Path
    image_index: int = 0

    def next_image_path(self, content_type: str | None) -> Path:
        self.image_index += 1
        extension = IMAGE_EXTENSIONS.get(content_type or "", ".bin")
        return self.image_dir / f"image_{self.image_index:04d}{extension}"


def parse_word_document(
    file_path: str | Path,
    *,
    image_analysis_workers: int = 3,
) -> list[dict[str, str]]:
    """Extract docx paragraphs, tables, and images in original order, then describe images."""
    started_at = time.perf_counter()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Word document not found: {path}")
    if path.suffix.lower() != ".docx":
        raise ValueError(f"Only .docx files are supported: {path}")

    _log(f"start parsing: {path}")
    document = Document(path)
    image_dir = Path("data") / "processing" / path.stem / "img"
    image_dir.mkdir(parents=True, exist_ok=True)

    stage_started_at = time.perf_counter()
    items = _extract_document_items(document, ParseContext(image_dir=image_dir))
    _log_item_summary("extracted document items", items, stage_started_at)

    stage_started_at = time.perf_counter()
    _log("start image analysis")
    items = enrich_image_descriptions(items, max_concurrency=image_analysis_workers)
    _log_item_summary("finished image analysis", items, stage_started_at)

    stage_started_at = time.perf_counter()
    _log("start link enrichment")
    items = enrich_links(items)
    _log_item_summary("finished link enrichment", items, stage_started_at)

    stage_started_at = time.perf_counter()
    output_path = write_items_to_txt(items, path)
    _log(f"wrote parsed txt: {output_path} ({time.perf_counter() - stage_started_at:.2f}s)")
    _log(f"finished parsing: {path.name} ({time.perf_counter() - started_at:.2f}s)")
    return items


def _extract_document_items(document: Any, context: ParseContext) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    table_index = 0
    paragraph_count = 0

    for block in _iter_blocks(document):
        if isinstance(block, Paragraph):
            paragraph_count += 1
            items.extend(_paragraph_items(block, source="paragraph", context=context))
        elif isinstance(block, Table):
            table_index += 1
            _log(f"extracting table {table_index}")
            table_started_at = time.perf_counter()
            items.extend(_table_items(block, table_index=table_index, context=context))
            _log(f"finished table {table_index} ({time.perf_counter() - table_started_at:.2f}s)")

    _log(f"scanned blocks: paragraphs={paragraph_count}, tables={table_index}, images={context.image_index}")
    return items


def _iter_blocks(parent: Any) -> Iterator[Paragraph | Table]:
    if hasattr(parent, "element") and hasattr(parent.element, "body"):
        parent_element = parent.element.body
    elif hasattr(parent, "_tc"):
        parent_element = parent._tc
    else:
        raise TypeError(f"Unsupported docx parent: {type(parent)!r}")

    for child in parent_element.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def _table_items(table: Table, table_index: int, context: ParseContext) -> list[dict[str, str]]:
    if _is_single_cell_table(table):
        return _plain_table_items(table, table_index=table_index, context=context)

    return _json_table_items(table, table_index=table_index, context=context)


def _plain_table_items(table: Table, table_index: int, context: ParseContext) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen_cells: set[int] = set()

    for row_index, row in enumerate(table.rows, start=1):
        for column_index, cell in enumerate(row.cells, start=1):
            cell_id = id(cell._tc)
            if cell_id in seen_cells:
                continue
            seen_cells.add(cell_id)

            source = f"table:{table_index}:row:{row_index}:column:{column_index}"
            nested_table_index = 0
            for block in _iter_blocks(cell):
                if isinstance(block, Paragraph):
                    items.extend(_paragraph_items(block, source=source, context=context))
                elif isinstance(block, Table):
                    nested_table_index += 1
                    nested_index = int(f"{table_index}{nested_table_index}")
                    items.extend(_table_items(block, table_index=nested_index, context=context))

    return items


def _is_single_cell_table(table: Table) -> bool:
    return len(table.rows) == 1 and bool(table.rows) and len(table.rows[0].cells) == 1


def _json_table_items(table: Table, table_index: int, context: ParseContext) -> list[dict[str, str]]:
    output_items: list[dict[str, str]] = []
    rows: list[list[str]] = []
    side_items: list[dict[str, str]] = []
    table_links: dict[str, str] = {}
    vertical_merge_values: dict[int, str] = {}
    table_width = _table_grid_width(table)

    def flush_table_rows() -> None:
        nonlocal rows, table_links
        table_items = _table_json_chunks(_table_object_rows(rows), table_index=table_index)
        for item in table_items:
            if table_links:
                item["links"] = dict(table_links)
        output_items.extend(table_items)
        rows = []
        table_links = {}

    for row_index, row in enumerate(table._tbl.tr_lst, start=1):
        detached_items = _detached_full_row_items(
            row,
            table,
            table_width=table_width,
            table_index=table_index,
            row_index=row_index,
            context=context,
        )
        if detached_items is not None:
            flush_table_rows()
            output_items.extend(detached_items)
            continue

        values: list[str] = []
        column_index = 1
        for tc in row.tc_lst:
            span = _cell_grid_span(tc)
            vertical_merge = _cell_vertical_merge(tc)
            source = f"table:{table_index}:row:{row_index}:column:{column_index}"
            if vertical_merge == "continue":
                text = vertical_merge_values.get(column_index, EMPTY_CELL_TEXT)
            else:
                cell_items = _cell_items(_Cell(tc, table), source=source, context=context)
                side_items.extend(item for item in cell_items if item.get("type") == "image")
                table_links.update(
                    {
                        item["description"]: item["url"]
                        for item in cell_items
                        if item.get("type") == "link_ref"
                        and item.get("description")
                        and item.get("url")
                    }
                )
                text = _cell_text(cell_items)
                if vertical_merge == "restart":
                    _set_vertical_merge_value(
                        vertical_merge_values,
                        column_index,
                        span,
                        text,
                    )

            values.extend([text] * span)
            column_index += span
        rows.append(values)

    flush_table_rows()
    return output_items + side_items


def _table_grid_width(table: Table) -> int:
    grid = table._tbl.tblGrid
    if grid is not None and grid.gridCol_lst:
        return len(grid.gridCol_lst)
    return max(
        (sum(_cell_grid_span(tc) for tc in row.tc_lst) for row in table._tbl.tr_lst),
        default=0,
    )


def _detached_full_row_items(
    row: Any,
    table: Table,
    *,
    table_width: int,
    table_index: int,
    row_index: int,
    context: ParseContext,
) -> list[dict[str, str]] | None:
    row_width = sum(_cell_grid_span(tc) for tc in row.tc_lst)
    if row_width < max(1, table_width):
        return None

    source = f"table:{table_index}:row:{row_index}:full"
    cells: list[tuple[str, list[dict[str, str]]]] = []
    for tc in row.tc_lst:
        cell_items = _cell_items(_Cell(tc, table), source=source, context=context)
        cells.append((_cell_text(cell_items), cell_items))

    meaningful_cells = [
        (text, cell_items)
        for text, cell_items in cells
        if text and text != EMPTY_CELL_TEXT
    ]
    if len(row.tc_lst) == 1 and meaningful_cells:
        text, cell_items = meaningful_cells[0]
    elif len(meaningful_cells) == 1 and len(meaningful_cells[0][0].strip()) > MAX_REPEATED_MERGED_CELL_CHARS:
        text, cell_items = meaningful_cells[0]
    else:
        return None

    output: list[dict[str, str]] = []
    output.append({"type": "paragraph", "style": "正文", "text": text, "source": source})
    output.extend(
        item for item in cell_items if item.get("type") in {"image", "link_ref"}
    )
    return output


def _cell_grid_span(tc: Any) -> int:
    tc_properties = tc.tcPr
    if tc_properties is None or tc_properties.gridSpan is None:
        return 1
    value = tc_properties.gridSpan.val
    return int(value) if value and str(value).isdigit() else 1


def _cell_vertical_merge(tc: Any) -> str | None:
    tc_properties = tc.tcPr
    if tc_properties is None or tc_properties.vMerge is None:
        return None
    return tc_properties.vMerge.val or "continue"


def _set_vertical_merge_value(
    vertical_merge_values: dict[int, str],
    column_index: int,
    span: int,
    text: str,
) -> None:
    if _should_repeat_merged_cell_text(text):
        for offset in range(span):
            vertical_merge_values[column_index + offset] = text
        return

    for offset in range(span):
        vertical_merge_values.pop(column_index + offset, None)


def _should_repeat_merged_cell_text(text: str) -> bool:
    return len(str(text).strip()) <= MAX_REPEATED_MERGED_CELL_CHARS


def _cell_items(cell: Any, source: str, context: ParseContext) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    nested_table_index = 0

    for block in _iter_blocks(cell):
        if isinstance(block, Paragraph):
            items.extend(_paragraph_items(block, source=source, context=context))
        elif isinstance(block, Table):
            nested_table_index += 1
            items.extend(_table_items(block, table_index=nested_table_index, context=context))

    return items


def _cell_text(items: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for item in items:
        item_type = item.get("type", "")
        if item_type == "image":
            parts.append(f"图片：{item.get('text', '')}")
        elif item_type == "link_ref":
            continue
        else:
            parts.append(item.get("text", ""))

    text = _clean_text(" ".join(part.strip() for part in parts if part.strip())).strip()
    return text or EMPTY_CELL_TEXT


def _table_object_rows(rows: list[list[str]]) -> list[dict[str, str]]:
    if len(rows) <= 1:
        return []

    headers = _unique_headers(_fill_empty_headers(rows[0]))
    object_rows: list[dict[str, str]] = []
    body_rows = rows[1:]
    for row in body_rows:
        normalized_row = _normalize_row(row, len(headers))
        object_rows.append(
            {
                header: normalized_row[index]
                for index, header in enumerate(headers)
            }
        )

    return object_rows


def _fill_empty_headers(headers: list[str]) -> list[str]:
    filled_headers: list[str] = []
    last_header = ""
    for index, header in enumerate(headers, start=1):
        normalized = header.strip()
        if normalized and normalized != EMPTY_CELL_TEXT:
            last_header = normalized
            filled_headers.append(normalized)
        elif last_header:
            filled_headers.append(last_header)
        else:
            filled_headers.append(f"column_{index}")
    return filled_headers


def _unique_headers(headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique_headers: list[str] = []
    for header in headers:
        counts[header] = counts.get(header, 0) + 1
        unique_headers.append(header if counts[header] == 1 else f"{header}_{counts[header]}")
    return unique_headers


def _normalize_row(row: list[str], width: int) -> list[str]:
    if len(row) >= width:
        return row[:width]
    return row + [EMPTY_CELL_TEXT] * (width - len(row))


def _table_json_chunks(rows: list[dict[str, str]], table_index: int) -> list[dict[str, str]]:
    if not rows:
        return []

    chunks: list[list[dict[str, str]]] = []
    current_rows: list[dict[str, str]] = []

    for row in rows:
        candidate = [*current_rows, row]
        if current_rows and len(_table_json_text(candidate)) > TABLE_JSON_MAX_CHARS:
            chunks.append(current_rows)
            current_rows = [row]
        else:
            current_rows.append(row)

    if current_rows:
        chunks.append(current_rows)

    return [
        {
            "type": "table",
            "style": "表格",
            "text": _table_json_text(chunk_rows),
            "source": f"table:{table_index}:part:{index}",
        }
        for index, chunk_rows in enumerate(chunks, start=1)
    ]


def _table_json_text(rows: list[dict[str, str]]) -> str:
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def _paragraph_items(paragraph: Paragraph, source: str, context: ParseContext) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    text_parts: list[str] = []
    style = _paragraph_style(paragraph)

    def flush_text() -> None:
        text = _clean_text("".join(text_parts)).strip()
        if text:
            items.append({"type": source, "style": style, "text": text})
        text_parts.clear()

    for child in paragraph._p.iterchildren():
        if child.tag == qn("w:r"):
            _append_run_items(
                Run(child, paragraph),
                paragraph,
                context,
                items,
                text_parts,
                flush_text,
                source,
            )
        elif child.tag == qn("w:hyperlink"):
            _append_hyperlink_items(child, paragraph, items, text_parts)

    flush_text()
    return items


def _append_run_items(
    run: Run,
    paragraph: Paragraph,
    context: ParseContext,
    items: list[dict[str, str]],
    text_parts: list[str],
    flush_text: Any,
    source: str,
) -> None:
    image_relationship_ids = _run_image_relationship_ids(run)
    if image_relationship_ids:
        flush_text()
        for relationship_id in image_relationship_ids:
            image_path = _save_image(paragraph, relationship_id, context)
            if image_path is not None:
                items.append(_image_item(image_path, source=source))

    if run.text:
        text_parts.append(_clean_text(run.text))


def _append_hyperlink_items(
    hyperlink: Any,
    paragraph: Paragraph,
    items: list[dict[str, str]],
    text_parts: list[str],
) -> None:
    display_text = _clean_text("".join(
        text_node.text or ""
        for text_node in hyperlink.iter(qn("w:t"))
    )).strip()
    relationship_id = hyperlink.get(qn("r:id"))
    target = ""
    if relationship_id:
        related_part = paragraph.part.related_parts.get(relationship_id)
        target = str(getattr(related_part, "target_ref", "") or "")

    if _is_dirty_ppt_link(display_text) or _is_dirty_ppt_link(target):
        return

    if target and display_text and target != display_text and not _looks_like_url(display_text):
        text_parts.append(display_text)
        items.append(
            {
                "type": "link_ref",
                "style": LINK_STYLE,
                "text": display_text,
                "url": target,
                "description": display_text,
            }
        )
        return

    if target:
        text_parts.append(target)
    elif display_text:
        text_parts.append(display_text)


def _clean_text(text: str) -> str:
    return PPT_LINK_PATTERN.sub("", text)


def _is_dirty_ppt_link(text: str) -> bool:
    return bool(text and PPT_LINK_PATTERN.search(text.strip()))


def _looks_like_url(text: str) -> bool:
    return bool(URL_TEXT_PATTERN.match(text.strip()))


def _paragraph_style(paragraph: Paragraph) -> str:
    if paragraph.style is not None and paragraph.style.name:
        return paragraph.style.name

    paragraph_properties = paragraph._p.pPr
    if paragraph_properties is not None:
        outline_level = _word_val(paragraph_properties.outlineLvl)
        if outline_level is not None and outline_level.isdigit():
            return f"标题 {int(outline_level) + 1}"

    if _looks_like_title(paragraph):
        return "标题"

    return "正文"


def _looks_like_title(paragraph: Paragraph) -> bool:
    max_size = 0
    has_bold = False

    for run in paragraph.runs:
        if run.bold is True or (run._r.rPr is not None and run._r.rPr.b is not None):
            has_bold = True
        if run._r.rPr is not None and run._r.rPr.sz is not None:
            size = _word_val(run._r.rPr.sz)
            if size is not None and size.isdigit():
                max_size = max(max_size, int(size))

    return has_bold and max_size >= 36


def _word_val(element: Any) -> str | None:
    if element is None:
        return None
    return element.get(qn("w:val"))


def _run_image_relationship_ids(run: Any) -> list[str]:
    relationship_ids: list[str] = []

    for blip in run._r.iter(qn("a:blip")):
        relationship_id = blip.get(qn("r:embed")) or blip.get(qn("r:link"))
        if relationship_id:
            relationship_ids.append(relationship_id)

    for image_data in run._r.iter(VML_IMAGE_DATA_TAG):
        relationship_id = image_data.get(qn("r:id"))
        if relationship_id:
            relationship_ids.append(relationship_id)

    return relationship_ids


def _save_image(paragraph: Paragraph, relationship_id: str, context: ParseContext) -> Path | None:
    related_part = paragraph.part.related_parts.get(relationship_id)
    if related_part is None or not getattr(related_part, "content_type", "").startswith("image/"):
        return None

    image_path = context.next_image_path(related_part.content_type)
    image_path.write_bytes(related_part.blob)
    return image_path


def _image_item(path: Path, source: str) -> dict[str, str]:
    return {
        "type": "image",
        "style": "图片",
        "source": source,
        "text": str(path),
        "path": str(path),
    }


def format_extracted_items(items: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for item in items:
        text = _format_item_text_for_txt(item["text"])
        if item["type"] == "image" and item.get("description"):
            text = f"{text}（{_format_item_text_for_txt(item['description'])}）"
        elif item["type"] in {"link", "link_ref"} and item.get("description"):
            text = _format_item_text_for_txt(item.get("url", text))
            text = f"{text}（{_format_item_text_for_txt(item['description'])}）"
        lines.append(f"[{item['type']}] [{item['style']}] {text}")
    return "\n".join(lines)


def _format_item_text_for_txt(text: str) -> str:
    return re.sub(r"\s*\r?\n\s*", " ", str(text)).strip()


def write_items_to_txt(items: list[dict[str, str]], source_file: str | Path) -> Path:
    source_path = Path(source_file)
    txt_dir = Path("data") / "processing" / source_path.stem / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    output_path = txt_dir / f"{source_path.stem}.txt"
    output_path.write_text(format_extracted_items(items), encoding="utf-8")
    return output_path


def _log(message: str) -> None:
    print(f"[word_parser] {message}", flush=True)


def _log_item_summary(message: str, items: list[dict[str, str]], started_at: float) -> None:
    counts = Counter(item.get("type", "unknown") for item in items)
    type_summary = ", ".join(f"{item_type}={count}" for item_type, count in sorted(counts.items()))
    _log(f"{message}: total={len(items)} ({type_summary or 'no items'}) ({time.perf_counter() - started_at:.2f}s)")


if __name__ == "__main__":
    target_file = Path("data") / "raw" / "info_or_data_query" / "管理动作频率表-学院.docx"
    extracted_items = parse_word_document(target_file)
    txt_path = Path("data") / "processing" / target_file.stem / "txt" / f"{target_file.stem}.txt"

    print(f"File: {target_file}")
    print(f"Extracted text blocks: {len(extracted_items)}")
    print(f"Text output: {txt_path}")
