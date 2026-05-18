from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from docx import Document
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

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


@dataclass
class ParseContext:
    image_dir: Path
    image_index: int = 0

    def next_image_path(self, content_type: str | None) -> Path:
        self.image_index += 1
        extension = IMAGE_EXTENSIONS.get(content_type or "", ".bin")
        return self.image_dir / f"image_{self.image_index:04d}{extension}"


def parse_word_document(file_path: str | Path) -> list[dict[str, str]]:
    """Extract docx paragraphs, tables, and images in original order, then describe images."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Word document not found: {path}")
    if path.suffix.lower() != ".docx":
        raise ValueError(f"Only .docx files are supported: {path}")

    document = Document(path)
    image_dir = Path("data") / "processing" / path.stem / "img"
    image_dir.mkdir(parents=True, exist_ok=True)

    items = _extract_document_items(document, ParseContext(image_dir=image_dir))
    items = enrich_image_descriptions(items)
    items = enrich_links(items)
    write_items_to_txt(items, path)
    return items


def _extract_document_items(document: Any, context: ParseContext) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    table_index = 0

    for block in _iter_blocks(document):
        if isinstance(block, Paragraph):
            items.extend(_paragraph_items(block, source="paragraph", context=context))
        elif isinstance(block, Table):
            table_index += 1
            items.extend(_table_items(block, table_index=table_index, context=context))

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


def _paragraph_items(paragraph: Paragraph, source: str, context: ParseContext) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    text_parts: list[str] = []
    style = _paragraph_style(paragraph)

    def flush_text() -> None:
        text = "".join(text_parts).strip()
        if text:
            items.append({"type": source, "style": style, "text": text})
        text_parts.clear()

    for run in paragraph.runs:
        image_relationship_ids = _run_image_relationship_ids(run)
        if image_relationship_ids:
            flush_text()
            for relationship_id in image_relationship_ids:
                image_path = _save_image(paragraph, relationship_id, context)
                if image_path is not None:
                    items.append(_image_item(image_path, source=source))

        if run.text:
            text_parts.append(run.text)

    flush_text()
    return items


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
        text = item["text"]
        if item["type"] == "image" and item.get("description"):
            text = f"{text}（{item['description']}）"
        elif item["type"] in {"link", "link_ref"} and item.get("description"):
            text = item.get("url", text)
            text = f"{text}（{item['description']}）"
        lines.append(f"[{item['type']}] [{item['style']}] {text}")
    return "\n".join(lines)


def write_items_to_txt(items: list[dict[str, str]], source_file: str | Path) -> Path:
    source_path = Path(source_file)
    txt_dir = Path("data") / "processing" / source_path.stem / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    output_path = txt_dir / f"{source_path.stem}.txt"
    output_path.write_text(format_extracted_items(items), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    target_file = Path("data") / "raw" / "process_guide" / "订阅号运营SOP.docx"
    extracted_items = parse_word_document(target_file)
    txt_path = Path("data") / "processing" / target_file.stem / "txt" / f"{target_file.stem}.txt"

    print(f"File: {target_file}")
    print(f"Extracted text blocks: {len(extracted_items)}")
    print(f"Text output: {txt_path}")
