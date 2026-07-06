from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

try:
    import fitz
except ModuleNotFoundError:  # pragma: no cover - exercised only in misconfigured envs
    fitz = None

try:
    from app.db.minio import (
        DEFAULT_RAW_DOCUMENT_BUCKET,
        build_minio_uri,
        parse_raw_document_reference,
        raw_document_object_name_for_file,
        upload_raw_document_file,
    )
    from app.services.data_clean import clean_items
    from app.services.parser.paths import processing_document_dir, processing_subdir
    from app.services.parser.pdf_parser import (
        PdfLine,
        _build_items as _build_pdf_items,
        _clean_pdf_lines,
        _extract_lines as _extract_pdf_lines,
    )
    from app.services.parser.word_parser import format_extracted_items
except ModuleNotFoundError:
    from app.db.minio import (
        DEFAULT_RAW_DOCUMENT_BUCKET,
        build_minio_uri,
        parse_raw_document_reference,
        raw_document_object_name_for_file,
        upload_raw_document_file,
    )
    from data_clean import clean_items
    from paths import processing_document_dir, processing_subdir
    from pdf_parser import (
        PdfLine,
        _build_items as _build_pdf_items,
        _clean_pdf_lines,
        _extract_lines as _extract_pdf_lines,
    )
    from word_parser import format_extracted_items


BODY_STYLE = "正文"
HEADING_STYLE = "标题"
DEFAULT_TITLE_PAGE_MAX_CHARS = 900
FONT_SIZE_TOLERANCE = 0.1
STANDARD_HYPHEN_CHARS = r"\-\u2010\u2011\u2012\u2013\u2014\u2212"
STANDARD_AST_PREFIX_PATTERN = r"S[ABF]\s*[" + STANDARD_HYPHEN_CHARS + r"]\s*\d+[A-Z]?"
STANDARD_INTERNATIONAL_SOURCE_PATTERN = r"(?:AS|CSA|EN|GB|IS|JIS)"
STANDARD_INTERNATIONAL_PATTERN = (
    r"S[AB]\s*/\s*"
    rf"{STANDARD_INTERNATIONAL_SOURCE_PATTERN}"
    r"(?:\s*[" + STANDARD_HYPHEN_CHARS + r"]\s*|\s+)"
    r"[A-Z]?\d[\w.\-]*"
)
STANDARD_CODE_PATTERN = re.compile(
    rf"\b(?:"
    rf"{STANDARD_AST_PREFIX_PATTERN}(?:\s*/\s*{STANDARD_AST_PREFIX_PATTERN})?|"
    rf"{STANDARD_INTERNATIONAL_PATTERN}"
    rf")\b",
    re.IGNORECASE,
)
STANDARD_CODE_LINE_PATTERN = re.compile(
    rf"^(?:"
    rf"{STANDARD_AST_PREFIX_PATTERN}(?:\s*/\s*{STANDARD_AST_PREFIX_PATTERN})?|"
    rf"{STANDARD_INTERNATIONAL_PATTERN}"
    rf")$",
    re.IGNORECASE,
)
FILENAME_UNSAFE_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
TITLE_PAGE_CODE_MIN_Y_RATIO = 0.12
TITLE_PAGE_CODE_MAX_Y_RATIO = 0.88


@dataclass(frozen=True)
class StandardTitleLine:
    text: str
    font_size: float
    y0: float
    order: int
    is_standard_code: bool


@dataclass(frozen=True)
class StandardTitlePage:
    page_number: int
    page_index: int
    title: str
    standard_code: str
    page_text_chars: int
    standard_code_count: int
    standard_font_size: float
    title_lines: list[StandardTitleLine]


@dataclass(frozen=True)
class StandardPdfSection:
    index: int
    title: str
    standard_code: str
    start_page: int
    end_page: int
    output_path: str
    section_dir: str = ""
    text_pdf_path: str = ""
    source_uri: str = ""


@dataclass(frozen=True)
class AssetCaption:
    asset_type: str
    text: str
    page_index: int
    bbox: tuple[float, float, float, float]
    font_size: float
    bold: bool


@dataclass(frozen=True)
class StandardAsset:
    index: int
    asset_type: str
    caption: str
    section_index: int
    standard_code: str
    section_page: int
    source_page: int
    bbox: tuple[float, float, float, float]
    image_path: str


TABLE_CAPTION_PATTERN = re.compile(
    r"^TABLE\s+(?:[A-Z]\d+(?:\.\d+)*|S\d+(?:\.\d+)*|\d+(?:\.\d+)*)(?:\b|[\s.,:;\-\u2013\u2014])"
)
FIGURE_CAPTION_PATTERN = re.compile(
    r"^FIG\.\s+(?:[A-Z]\d+(?:\.\d+)*|S\d+(?:\.\d+)*|\d+(?:\.\d+)*)(?:\b|[\s.,:;\-\u2013\u2014])"
)
CONTINUED_PATTERN = re.compile(r"^Continued$", re.IGNORECASE)
CAPTION_CODE_ONLY_PATTERN = re.compile(
    r"^(?:TABLE|FIG\.)\s+(?:[A-Z]\d+(?:\.\d+)*|S\d+(?:\.\d+)*|\d+(?:\.\d+)*)$"
)
BODY_TOP_MARGIN = 45.0
BODY_BOTTOM_MARGIN = 45.0
BODY_SIDE_MARGIN = 45.0
CAPTION_ROW_TOLERANCE = 3.0
CAPTION_CONTINUED_GAP = 80.0
CAPTION_SAME_ROW_GAP = 240.0
CAPTION_CONTINUATION_GAP = 16.0
CAPTION_CENTER_TOLERANCE = 150.0
CAPTION_MAX_PARTS = 5
TABLE_DRAWING_START_GAP = 180.0
TABLE_DRAWING_CHAIN_GAP = 55.0
TABLE_TEXT_CHAIN_GAP = 24.0
TABLE_NOTE_MAX_GAP = 140.0
TABLE_NOTE_MAX_FONT_SIZE = 8.4
IMPLICIT_TABLE_ROW_TOLERANCE = 3.0
IMPLICIT_TABLE_MAX_FONT_SIZE = 8.7
IMPLICIT_TABLE_MAX_ROW_GAP = 6.0
MASK_EXTRA_NOTE_GAP = 90.0
ASSET_RENDER_ZOOM = 2.0
STANDARD_TEXT_HEADING_FONT_TOLERANCE = 0.3
STANDARD_TEXT_CONTINUATION_GAP_MULTIPLIER = 1.8
STANDARD_TEXT_MIN_TITLE_FONT_SIZE = 10.0
STANDARD_TEXT_FIRST_TITLE_SIZE_DELTA = 5.0
STANDARD_TEXT_LEFT_TOLERANCE = 8.0
STANDARD_TEXT_CENTER_MAX_CHARS = 180
STANDARD_TEXT_NUMBERED_MAX_CHARS = 180
STANDARD_NUMBERED_HEADING_PATTERN = re.compile(r"^(\d{1,3}(?:\.\d{1,3})*)\.?\s+(.+)")
STANDARD_NUMBER_ONLY_HEADING_PATTERN = re.compile(r"^(\d{1,3}(?:\.\d{1,3})*)\.$")
STANDARD_REFERENCED_DOCUMENTS_HEADING_PATTERN = re.compile(
    r"^\d{1,3}\.?\s+Referenced\s+Documents\s*$",
    re.IGNORECASE,
)
STANDARD_SMALL_RESIDUAL_FONT_DELTA = 0.7
STANDARD_SMALL_RESIDUAL_MAX_CHARS = 180


def parse_standard_pdf_document(
    file_path: str | Path,
    *,
    image_analysis_workers: int = 3,
    title_page_max_chars: int = DEFAULT_TITLE_PAGE_MAX_CHARS,
    source_reference: str | Path | None = None,
    split_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Split long ASME/ASTM standard PDFs into smaller PDF sections.

    This parser is intentionally focused on the first debugging step: detecting
    standard title pages and writing per-standard PDF chunks. Full text/table
    extraction can be layered onto these section files after the boundary logic
    is stable.
    """
    sections = split_standard_pdf_document(
        file_path,
        image_analysis_workers=image_analysis_workers,
        title_page_max_chars=title_page_max_chars,
        source_reference=source_reference,
        split_prefix=split_prefix,
    )
    items, txt_paths = extract_and_write_standard_section_texts(sections)

    _log(f"wrote parsed txt files: {len(txt_paths)}")
    return items


def split_standard_pdf_document(
    file_path: str | Path,
    *,
    image_analysis_workers: int = 3,
    title_page_max_chars: int = DEFAULT_TITLE_PAGE_MAX_CHARS,
    source_reference: str | Path | None = None,
    split_prefix: str | None = None,
) -> list[StandardPdfSection]:
    """Split a standard PDF and publish section PDFs under the raw MinIO bucket."""
    del image_analysis_workers
    started_at = time.perf_counter()
    _ensure_dependencies()

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF document not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Only .pdf files are supported: {path}")
    if path.name.startswith("~$"):
        raise ValueError(f"Temporary PDF lock files are not supported: {path}")

    _log(f"start splitting: {path}")
    title_pages = find_standard_title_pages(path, title_page_max_chars=title_page_max_chars)
    _log(f"detected standard title pages: {len(title_pages)}")

    sections = split_standard_pdf(path, title_pages)
    sections = upload_standard_pdf_sections_to_minio(
        path,
        sections,
        source_reference=source_reference,
        split_prefix=split_prefix,
    )
    assets_manifest_path = extract_standard_assets_from_sections(path, sections)
    assets = load_standard_assets_manifest(assets_manifest_path)
    sections = write_masked_text_pdfs(path, sections, assets)
    manifest_path = write_split_manifest(
        path,
        title_pages,
        sections,
        title_page_max_chars=title_page_max_chars,
        source_reference=source_reference,
    )

    _log(f"wrote sections: {len(sections)}")
    _log(f"wrote manifest: {manifest_path}")
    _log(f"wrote assets manifest: {assets_manifest_path}")
    _log("wrote masked text PDFs")
    _log(f"finished splitting: {path.name} ({time.perf_counter() - started_at:.2f}s)")
    return sections


def find_standard_title_pages(
    file_path: str | Path,
    *,
    title_page_max_chars: int = DEFAULT_TITLE_PAGE_MAX_CHARS,
) -> list[StandardTitlePage]:
    _ensure_fitz()
    path = Path(file_path)
    title_pages: list[StandardTitlePage] = []
    with fitz.open(path) as document:
        last_page_index = len(document) - 1
        for page_index, page in enumerate(document):
            if page_index in {0, last_page_index}:
                continue
            candidate = _title_page_candidate(
                page,
                page_index=page_index,
                title_page_max_chars=title_page_max_chars,
            )
            if candidate is not None:
                title_pages.append(candidate)
    return title_pages


def split_standard_pdf(file_path: str | Path, title_pages: list[StandardTitlePage]) -> list[StandardPdfSection]:
    _ensure_fitz()
    path = Path(file_path)
    output_dir = processing_document_dir(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_existing_standard_outputs(output_dir)

    sections: list[StandardPdfSection] = []
    used_names: set[str] = set()

    with fitz.open(path) as source_document:
        page_count = len(source_document)
        content_start_index = 1 if page_count > 1 else 0
        content_end_index = page_count - 2 if page_count > 1 else page_count - 1
        if content_end_index < content_start_index:
            return []

        sorted_titles = sorted(title_pages, key=lambda item: item.page_index)
        ranges = _section_ranges(sorted_titles, content_start_index, content_end_index)

        for index, (title_page, start_index, end_index) in enumerate(ranges, start=1):
            title = title_page.title if title_page is not None else path.stem
            standard_code = title_page.standard_code if title_page is not None else ""
            section_name = _unique_name(_section_base_name(title, standard_code), used_names)
            section_dir = output_dir / section_name
            pdf_dir = section_dir / "pdf"
            pdf_dir.mkdir(parents=True, exist_ok=True)
            output_path = pdf_dir / f"{section_name}.pdf"

            with fitz.open() as section_document:
                section_document.insert_pdf(source_document, from_page=start_index, to_page=end_index)
                section_document.save(output_path, garbage=4, deflate=True)

            sections.append(
                StandardPdfSection(
                    index=index,
                    title=title,
                    standard_code=standard_code,
                    start_page=start_index + 1,
                    end_page=end_index + 1,
                    output_path=str(output_path),
                    section_dir=str(section_dir),
                )
            )
    return sections


def upload_standard_pdf_sections_to_minio(
    file_path: str | Path,
    sections: list[StandardPdfSection],
    *,
    source_reference: str | Path | None = None,
    bucket: str | None = None,
    split_prefix: str | None = None,
) -> list[StandardPdfSection]:
    if not sections:
        return []

    source_ref = source_reference if source_reference is not None else file_path
    reference = _standard_source_reference(source_ref, bucket=bucket)
    target_bucket = bucket or reference.bucket or DEFAULT_RAW_DOCUMENT_BUCKET
    target_prefix = _normalize_split_prefix(split_prefix) if split_prefix else _standard_split_object_prefix(reference.object_name)

    uploaded: list[StandardPdfSection] = []
    for section in sections:
        section_path = Path(section.output_path)
        object_name = f"{target_prefix}/{section_path.name}"
        upload_raw_document_file(
            section_path,
            bucket=target_bucket,
            object_name=object_name,
        )
        uploaded.append(
            replace(
                section,
                source_uri=build_minio_uri(target_bucket, object_name),
            )
        )
    return uploaded


def write_split_manifest(
    file_path: str | Path,
    title_pages: list[StandardTitlePage],
    sections: list[StandardPdfSection],
    *,
    title_page_max_chars: int = DEFAULT_TITLE_PAGE_MAX_CHARS,
    source_reference: str | Path | None = None,
) -> Path:
    source_path = Path(file_path)
    output_dir = processing_document_dir(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    payload = {
        "source_pdf": str(source_path),
        "source_reference": str(source_reference or file_path),
        "title_page_max_chars": title_page_max_chars,
        "cover_policy": "skip first and last source pages",
        "detected_title_pages": [_title_page_to_dict(item) for item in title_pages],
        "sections": [asdict(item) for item in sections],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def extract_standard_assets_from_sections(
    source_file: str | Path,
    sections: list[StandardPdfSection],
    *,
    render_zoom: float = ASSET_RENDER_ZOOM,
) -> Path:
    source_path = Path(source_file)
    output_dir = processing_document_dir(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_legacy_asset_dir(output_dir)
    _clear_existing_section_images(source_path, sections)

    assets: list[StandardAsset] = []
    used_asset_names: dict[Path, set[str]] = {}
    for section in sections:
        section_path = Path(section.output_path)
        if not section_path.exists():
            continue
        image_dir = _section_image_dir(source_path, section)
        image_dir.mkdir(parents=True, exist_ok=True)
        used_names = used_asset_names.setdefault(image_dir, set())
        with fitz.open(section_path) as document:
            for page_index, page in enumerate(document):
                captions = _asset_captions_from_page(page, page_index=page_index)
                for caption_index, caption in enumerate(captions):
                    clip = _asset_clip_for_caption(page, captions, caption_index)
                    if clip is None or clip.is_empty or clip.width <= 1 or clip.height <= 1:
                        continue
                    asset_index = len(assets) + 1
                    filename = _unique_filename(_asset_filename(caption), used_names)
                    image_path = image_dir / filename
                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(render_zoom, render_zoom),
                        clip=clip,
                        alpha=False,
                    )
                    pixmap.save(image_path)
                    assets.append(
                        StandardAsset(
                            index=asset_index,
                            asset_type=caption.asset_type,
                            caption=caption.text,
                            section_index=section.index,
                            standard_code=section.standard_code,
                            section_page=page_index + 1,
                            source_page=section.start_page + page_index,
                            bbox=_rect_tuple(clip),
                            image_path=str(image_path),
                        )
                    )

    manifest_path = output_dir / "assets_manifest.json"
    payload = {
        "source_pdf": str(source_path),
        "render_zoom": render_zoom,
        "asset_count": len(assets),
        "assets": [asdict(asset) for asset in assets],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def load_standard_assets_manifest(manifest_path: str | Path) -> list[StandardAsset]:
    path = Path(manifest_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [StandardAsset(**item) for item in payload.get("assets", [])]


def write_masked_text_pdfs(
    source_file: str | Path,
    sections: list[StandardPdfSection],
    assets: list[StandardAsset],
) -> list[StandardPdfSection]:
    """Write per-section PDFs with detected assets redacted for text extraction."""
    source_path = Path(source_file)
    _clear_existing_section_text_pdfs(source_path, sections)
    assets_by_section_page = _assets_by_section_page(assets)
    output: list[StandardPdfSection] = []

    for section in sections:
        section_path = Path(section.output_path)
        text_pdf_dir = _section_text_pdf_dir(source_path, section)
        text_pdf_dir.mkdir(parents=True, exist_ok=True)
        text_pdf_path = text_pdf_dir / f"{section_path.stem}.masked.pdf"
        if not section_path.exists():
            output.append(replace(section, text_pdf_path=str(text_pdf_path)))
            continue

        with fitz.open(section_path) as document:
            page_rects = assets_by_section_page.get(section.index, {})
            for page_index, page in enumerate(document):
                page_number = page_index + 1
                rects = page_rects.get(page_number, [])
                mask_rects = [*_expanded_asset_mask_rects(page, rects), *_implicit_table_rects(page)]
                for rect in _merged_mask_rects(mask_rects, page.rect):
                    redact_rect = _redaction_rect_for_page(page, rect)
                    if redact_rect.is_empty or redact_rect.width <= 1 or redact_rect.height <= 1:
                        continue
                    page.add_redact_annot(redact_rect, fill=(1, 1, 1))
                if mask_rects:
                    page.apply_redactions(
                        images=fitz.PDF_REDACT_IMAGE_PIXELS,
                        graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                        text=fitz.PDF_REDACT_TEXT_REMOVE,
                    )
            document.save(text_pdf_path, garbage=4, deflate=True)
        output.append(replace(section, text_pdf_path=str(text_pdf_path)))
    return output


def extract_standard_text_from_sections(sections: list[StandardPdfSection]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for section in sections:
        items.extend(extract_standard_text_from_section(section))
    return items


def extract_and_write_standard_section_texts(sections: list[StandardPdfSection]) -> tuple[list[dict[str, Any]], list[Path]]:
    all_items: list[dict[str, Any]] = []
    txt_paths: list[Path] = []
    for section in sections:
        items, referenced_items = extract_standard_text_payload_from_section(section)
        all_items.extend(items)
        txt_paths.append(write_section_items_to_txt(items, section))
        if referenced_items:
            write_section_referenced_documents_to_json(referenced_items, section)
    return all_items, txt_paths


def extract_standard_text_from_section(section: StandardPdfSection) -> list[dict[str, Any]]:
    items, _referenced_items = extract_standard_text_payload_from_section(section)
    return items


def extract_standard_text_payload_from_section(
    section: StandardPdfSection,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lines: list[Any] = []
    referenced_lines: list[Any] = []
    text_source = Path(section.text_pdf_path or section.output_path)
    if not text_source.exists():
        return [], []

    section_lines, section_referenced_lines = _standard_section_text_lines(_extract_pdf_lines(text_source))
    for next_order, line in enumerate(section_lines):
        line.page_number = section.start_page + line.page_number - 1
        line.order = next_order
        lines.append(line)

    for next_order, line in enumerate(section_referenced_lines):
        line.page_number = section.start_page + line.page_number - 1
        line.order = next_order
        referenced_lines.append(line)

    return clean_items(_build_pdf_items(lines)), clean_items(_build_pdf_items(referenced_lines))


def _standard_section_text_lines(lines: list[PdfLine]) -> tuple[list[PdfLine], list[PdfLine]]:
    """Assign ASME/ASTM standard-specific heading styles and merge wrapped headings."""
    if not lines:
        return [], []

    ordered = _standard_ordered_lines(lines)
    first_page = min(line.page_number for line in ordered)
    title_line, consumed_ids = _standard_first_page_title_line(ordered, first_page=first_page)
    ordered = _standard_filter_small_residual_lines(_standard_ordered_lines(_clean_pdf_lines(ordered)))
    center_heading_sizes = _standard_center_heading_sizes(ordered, first_page=first_page)

    output: list[PdfLine] = []
    if title_line is not None:
        output.append(title_line)

    index = 0
    while index < len(ordered):
        line = ordered[index]
        if id(line) in consumed_ids:
            index += 1
            continue

        kind = _standard_heading_kind(line, first_page=first_page, center_heading_sizes=center_heading_sizes)
        if kind is None:
            output.append(replace(line, style=BODY_STYLE))
            index += 1
            continue

        group = [line]
        index += 1
        while index < len(ordered):
            next_line = ordered[index]
            if id(next_line) in consumed_ids:
                index += 1
                continue
            if not _is_standard_heading_continuation(group[-1], next_line, kind):
                break
            group.append(next_line)
            index += 1

        output.append(_merge_standard_text_lines(group, style=_standard_heading_style(kind)))

    return _split_referenced_documents_section(output)


def _standard_ordered_lines(lines: list[PdfLine]) -> list[PdfLine]:
    return sorted(lines, key=_standard_line_order_key)


def _standard_filter_small_residual_lines(lines: list[PdfLine]) -> list[PdfLine]:
    body_font_size = _standard_body_font_size(lines)
    if body_font_size <= 0:
        return lines
    threshold = body_font_size - STANDARD_SMALL_RESIDUAL_FONT_DELTA
    return [line for line in lines if not _looks_like_small_asset_residual(line, threshold)]


def _standard_body_font_size(lines: list[PdfLine]) -> float:
    sizes = [
        float(line.font_size or 0.0)
        for line in lines
        if line.font_size > 0
        and not line.bold
        and 15 <= _text_length(line.text) <= 240
        and line.alignment != "center"
        and not _is_standard_code_line(line.text)
    ]
    return float(median(sizes)) if sizes else 0.0


def _looks_like_small_asset_residual(line: PdfLine, threshold: float) -> bool:
    if line.bold or line.font_size > threshold:
        return False
    text = _clean_line_text(line.text)
    if not text or _text_length(text) > STANDARD_SMALL_RESIDUAL_MAX_CHARS:
        return False
    upper = text.upper()
    if TABLE_CAPTION_PATTERN.match(upper) or FIGURE_CAPTION_PATTERN.match(upper):
        return True
    if CONTINUED_PATTERN.match(text) or CAPTION_CODE_ONLY_PATTERN.match(upper):
        return True
    if re.match(r"^(?:NOTE|NOTES|FOOTNOTE|SOURCE)\b", text, re.IGNORECASE):
        return True
    if re.match(r"^[A-Z](?:\s|[.,;:])", text) and len(text) <= 120:
        return True
    if re.match(r"^\(?[a-z]\)?\s+", text) and len(text) <= 120:
        return True
    return False


def _standard_line_order_key(line: PdfLine) -> tuple[int, int, float, float, int]:
    page_width = max(1.0, float(line.page_width or 0.0))
    full_width = (line.x0 <= page_width * 0.22 and line.x1 >= page_width * 0.78) or (
        line.alignment == "center" and (line.x1 - line.x0) >= page_width * 0.45
    ) or (
        line.alignment == "center" and line.bold and line.font_size >= STANDARD_TEXT_MIN_TITLE_FONT_SIZE
    )
    if full_width:
        column = 0
    elif line.x0 >= page_width * 0.48:
        column = 2
    else:
        column = 1
    return (line.page_number, column, line.y0, line.x0, line.order)


def _standard_first_page_title_line(lines: list[PdfLine], *, first_page: int) -> tuple[PdfLine | None, set[int]]:
    first_page_lines = [
        line
        for line in lines
        if line.page_number == first_page
        and line.bold
        and line.font_size >= STANDARD_TEXT_MIN_TITLE_FONT_SIZE
        and _is_body_title_area_line(line)
        and _text_length(line.text) <= STANDARD_TEXT_CENTER_MAX_CHARS
    ]
    if not first_page_lines:
        return None, set()

    max_size = max(line.font_size for line in first_page_lines)
    title_parts = [
        line
        for line in first_page_lines
        if line.font_size >= max_size - STANDARD_TEXT_HEADING_FONT_TOLERANCE
        or (
            _is_standard_code_line(line.text)
            and line.font_size >= max(STANDARD_TEXT_MIN_TITLE_FONT_SIZE, max_size - STANDARD_TEXT_FIRST_TITLE_SIZE_DELTA)
        )
    ]
    if not title_parts:
        return None, set()

    title_parts = sorted(title_parts, key=lambda item: (item.y0, item.x0, item.order))
    return _merge_standard_text_lines(title_parts, style=HEADING_STYLE), {id(line) for line in title_parts}


def _standard_center_heading_sizes(lines: list[PdfLine], *, first_page: int) -> dict[int, float]:
    output: dict[int, float] = {}
    for line in lines:
        if line.page_number == first_page:
            continue
        if not _is_standard_center_heading_candidate(line):
            continue
        output[line.page_number] = max(output.get(line.page_number, 0.0), float(line.font_size or 0.0))
    return output


def _standard_heading_kind(
    line: PdfLine,
    *,
    first_page: int,
    center_heading_sizes: dict[int, float],
) -> tuple[str, int] | None:
    if line.page_number != first_page and _is_standard_center_heading_candidate(line):
        page_center_size = center_heading_sizes.get(line.page_number, 0.0)
        if page_center_size and line.font_size >= page_center_size - STANDARD_TEXT_HEADING_FONT_TOLERANCE:
            return ("center", 2)

    number_level = _standard_numbered_heading_level(line)
    if number_level is not None:
        return ("numbered", number_level)
    return None


def _is_standard_center_heading_candidate(line: PdfLine) -> bool:
    return (
        line.alignment == "center"
        and line.bold
        and line.font_size >= STANDARD_TEXT_MIN_TITLE_FONT_SIZE
        and not _is_standard_code_line(line.text)
        and _text_length(line.text) <= STANDARD_TEXT_CENTER_MAX_CHARS
    )


def _standard_numbered_heading_level(line: PdfLine) -> int | None:
    text = line.text.strip()
    if _looks_like_parenthesized_number_marker(text):
        return None
    match = STANDARD_NUMBERED_HEADING_PATTERN.match(text)
    if match is None:
        match = STANDARD_NUMBER_ONLY_HEADING_PATTERN.match(text)
    if not match:
        return None
    if not line.bold or line.alignment == "right":
        return None
    if _text_length(text) > STANDARD_TEXT_NUMBERED_MAX_CHARS:
        return None
    if "." in match.group(1):
        return None
    if STANDARD_NUMBER_ONLY_HEADING_PATTERN.match(text):
        return 3
    return 3


def _looks_like_parenthesized_number_marker(text: str) -> bool:
    return bool(re.match(r"^[\(（]\s*\d{1,3}\s*[\)）](?:\s|$)", text.strip()))


def _is_standard_heading_continuation(previous: PdfLine, current: PdfLine, kind: tuple[str, int]) -> bool:
    if current.page_number != previous.page_number:
        return False
    if kind[0] == "numbered" and STANDARD_NUMBER_ONLY_HEADING_PATTERN.match(previous.text.strip()):
        if STANDARD_NUMBERED_HEADING_PATTERN.match(current.text.strip()):
            return False
        if _text_length(current.text) > 80:
            return False
        if abs(current.font_size - previous.font_size) > STANDARD_TEXT_HEADING_FONT_TOLERANCE:
            return False
        if current.y0 - previous.y1 > max(8.0, previous.font_size * STANDARD_TEXT_CONTINUATION_GAP_MULTIPLIER):
            return False
        if abs(current.x0 - previous.x0) <= STANDARD_TEXT_LEFT_TOLERANCE:
            return True
        same_row = abs(current.y0 - previous.y0) <= STANDARD_TEXT_HEADING_FONT_TOLERANCE
        return same_row and 0 <= current.x0 - previous.x1 <= 120.0
    if current.bold != previous.bold:
        return False
    if abs(current.font_size - previous.font_size) > STANDARD_TEXT_HEADING_FONT_TOLERANCE:
        return False
    if current.y0 - previous.y1 > max(8.0, previous.font_size * STANDARD_TEXT_CONTINUATION_GAP_MULTIPLIER):
        return False
    if kind[0] == "numbered":
        if STANDARD_NUMBERED_HEADING_PATTERN.match(current.text.strip()):
            return False
        return abs(current.x0 - previous.x0) <= STANDARD_TEXT_LEFT_TOLERANCE
    return current.alignment == "center"


def _standard_heading_style(kind: tuple[str, int]) -> str:
    level = max(1, min(kind[1], 6))
    return HEADING_STYLE if level == 1 else f"{HEADING_STYLE} {level}"


def _split_referenced_documents_section(lines: list[PdfLine]) -> tuple[list[PdfLine], list[PdfLine]]:
    output: list[PdfLine] = []
    referenced_documents: list[PdfLine] = []
    skip_level: int | None = None

    for line in lines:
        level = _standard_line_heading_level(line)
        if skip_level is not None:
            if level is not None and level <= skip_level:
                skip_level = None
            else:
                referenced_documents.append(line)
                continue

        if _is_referenced_documents_heading(line):
            skip_level = level or _standard_numbered_heading_level(line) or 3
            referenced_documents.append(line)
            continue

        output.append(line)
    return output, referenced_documents


def _is_referenced_documents_heading(line: PdfLine) -> bool:
    if _standard_line_heading_level(line) is None:
        return False
    return bool(STANDARD_REFERENCED_DOCUMENTS_HEADING_PATTERN.match(_clean_line_text(line.text)))


def _standard_line_heading_level(line: PdfLine) -> int | None:
    style = str(line.style or "")
    if style == HEADING_STYLE:
        return 1
    if not style.startswith(f"{HEADING_STYLE} "):
        return None
    match = re.search(r"\d+", style)
    return int(match.group(0)) if match else None


def _merge_standard_text_lines(lines: list[PdfLine], *, style: str) -> PdfLine:
    if len(lines) == 1:
        return replace(lines[0], text=_clean_line_text(lines[0].text), style=style)
    ordered = sorted(lines, key=lambda item: (item.y0, item.x0, item.order))
    return PdfLine(
        text=_clean_line_text(" ".join(line.text for line in ordered)),
        page_number=ordered[0].page_number,
        page_width=ordered[0].page_width,
        page_height=ordered[0].page_height,
        x0=min(line.x0 for line in ordered),
        y0=min(line.y0 for line in ordered),
        x1=max(line.x1 for line in ordered),
        y1=max(line.y1 for line in ordered),
        font_size=max(line.font_size for line in ordered),
        bold=any(line.bold for line in ordered),
        order=min(line.order for line in ordered),
        alignment=ordered[0].alignment,
        style=style,
        urls=[],
    )


def _is_body_title_area_line(line: PdfLine) -> bool:
    if line.page_height <= 0:
        return True
    return TITLE_PAGE_CODE_MIN_Y_RATIO * line.page_height <= line.y0 <= TITLE_PAGE_CODE_MAX_Y_RATIO * line.page_height


def _assets_by_section_page(assets: list[StandardAsset]) -> dict[int, dict[int, list[Any]]]:
    output: dict[int, dict[int, list[Any]]] = {}
    for asset in assets:
        output.setdefault(asset.section_index, {}).setdefault(asset.section_page, []).append(fitz.Rect(asset.bbox))
    return output


def _merged_mask_rects(rects: list[Any], page_rect: Any) -> list[Any]:
    masks = sorted(
        (_clamp_rect(_expanded_rect(rect, 1.0), page_rect) for rect in rects),
        key=lambda item: (item.y0, item.x0),
    )
    output: list[Any] = []
    for rect in masks:
        if rect.is_empty or rect.width <= 1 or rect.height <= 1:
            continue
        for existing in output:
            if existing.intersects(rect) or (
                _horizontal_overlap(existing, rect) > 0 and _rects_near_vertically(existing, rect, max_gap=2)
            ):
                existing.include_rect(rect)
                break
        else:
            output.append(fitz.Rect(rect))
    return output


def _redaction_rect_for_page(page: Any, rect: Any) -> Any:
    output = fitz.Rect(rect)
    if int(getattr(page, "rotation", 0) or 0) != 0:
        output = output * page.derotation_matrix
        output.intersect(page.mediabox)
    else:
        output.intersect(page.rect)
    return output


def _expanded_asset_mask_rects(page: Any, rects: list[Any]) -> list[Any]:
    output: list[Any] = []
    text_items = _page_text_items(page)
    for rect in rects:
        mask = fitz.Rect(rect)
        for note_rect in _small_note_rects_after(mask, text_items):
            mask.include_rect(note_rect)
        output.append(mask)
    return output


def _small_note_rects_after(mask_rect: Any, text_items: list[dict[str, Any]]) -> list[Any]:
    notes: list[Any] = []
    current = fitz.Rect(mask_rect)
    for item in sorted(text_items, key=lambda value: (value["rect"].y0, value["rect"].x0)):
        rect = item["rect"]
        if rect.y0 < current.y1 - 1:
            continue
        if rect.y0 - current.y1 > MASK_EXTRA_NOTE_GAP:
            break
        if _horizontal_overlap(rect, current) <= 1:
            continue
        if not _looks_like_table_note_item(item):
            if notes:
                break
            continue
        notes.append(rect)
        current.include_rect(rect)
    return notes


def _implicit_table_rects(page: Any) -> list[Any]:
    rows = _implicit_table_candidate_rows(page)
    if not rows:
        return []

    output: list[Any] = []
    current: list[list[dict[str, Any]]] = []
    previous_rect: Any | None = None

    def flush() -> None:
        if len(current) < 3:
            current.clear()
            return
        rects = [item["rect"] for row in current for item in row]
        output.append(_expanded_rect(_union_rects(rects), 4.0))
        current.clear()

    for row in rows:
        row_rect = _union_rects(item["rect"] for item in row)
        if previous_rect is not None and row_rect.y0 - previous_rect.y1 > IMPLICIT_TABLE_MAX_ROW_GAP:
            flush()
        current.append(row)
        previous_rect = row_rect
    flush()
    return output


def _implicit_table_candidate_rows(page: Any) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for item in sorted(_page_text_items(page), key=lambda value: (value["rect"].y0, value["rect"].x0)):
        if not _looks_like_implicit_table_cell(item):
            continue
        rect = item["rect"]
        for row in rows:
            if abs(rect.y0 - row[0]["rect"].y0) <= IMPLICIT_TABLE_ROW_TOLERANCE:
                row.append(item)
                break
        else:
            rows.append([item])

    output: list[list[dict[str, Any]]] = []
    for row in rows:
        cells = sorted(row, key=lambda value: value["rect"].x0)
        if len(cells) < 2:
            continue
        if _row_has_distinct_columns(cells):
            output.append(cells)
    return output


def _looks_like_implicit_table_cell(item: dict[str, Any]) -> bool:
    text = str(item.get("text") or "").strip()
    font_size = float(item.get("font_size") or 0)
    if not text or font_size <= 0 or font_size > IMPLICIT_TABLE_MAX_FONT_SIZE:
        return False
    if len(text) > 120:
        return False
    return True


def _row_has_distinct_columns(cells: list[dict[str, Any]]) -> bool:
    previous = cells[0]["rect"]
    for cell in cells[1:]:
        rect = cell["rect"]
        if rect.x0 - previous.x1 >= 18:
            return True
        previous = rect
    return False


def _expanded_rect(rect: Any, amount: float) -> Any:
    output = fitz.Rect(rect)
    output.x0 -= amount
    output.y0 -= amount
    output.x1 += amount
    output.y1 += amount
    return output


def _asset_captions_from_page(page: Any, *, page_index: int) -> list[AssetCaption]:
    lines = _asset_lines_from_page(page, page_index=page_index)
    captions: list[AssetCaption] = []
    for index, line in enumerate(lines):
        caption_type = _caption_type(line["text"])
        if caption_type is None:
            continue
        if not line["bold"] and not line["text"].startswith(("TABLE ", "FIG. ")):
            continue
        text, bbox = _caption_text_and_bbox(lines, index)
        captions.append(
            AssetCaption(
                asset_type=caption_type,
                text=text,
                page_index=page_index,
                bbox=_rect_tuple(bbox),
                font_size=line["font_size"],
                bold=line["bold"],
            )
        )
    return captions


def _asset_lines_from_page(page: Any, *, page_index: int) -> list[dict[str, Any]]:
    del page_index
    lines: list[dict[str, Any]] = []
    for block in page.get_text("dict", sort=True).get("blocks", []):
        if block.get("type") != 0:
            continue
        for raw_line in block.get("lines", []):
            spans = raw_line.get("spans", [])
            text_parts: list[str] = []
            sizes: list[float] = []
            bold = False
            boxes: list[tuple[float, float, float, float]] = []
            for span in spans:
                text = _clean_line_text(str(span.get("text") or ""))
                if not text:
                    continue
                text_parts.append(text)
                size = float(span.get("size") or 0)
                if size > 0:
                    sizes.append(size)
                font = str(span.get("font") or "")
                flags = int(span.get("flags") or 0)
                bold = bold or _is_bold_font(font, flags)
                bbox = span.get("bbox") or (0, 0, 0, 0)
                boxes.append(tuple(float(value) for value in bbox))
            text = _clean_line_text(" ".join(text_parts))
            if not text or not boxes:
                continue
            rect = _union_rects(fitz.Rect(box) for box in boxes)
            lines.append(
                {
                    "text": text,
                    "bbox": _rect_tuple(rect),
                    "font_size": max(sizes) if sizes else 0.0,
                    "bold": bold,
                    "page_width": float(page.rect.width),
                }
            )
    return sorted(lines, key=lambda item: (fitz.Rect(item["bbox"]).y0, fitz.Rect(item["bbox"]).x0))


def _caption_type(text: str) -> str | None:
    value = _clean_line_text(text)
    if TABLE_CAPTION_PATTERN.match(value):
        return "table"
    if FIGURE_CAPTION_PATTERN.match(value):
        return "figure"
    return None


def _caption_text_and_bbox(lines: list[dict[str, Any]], start_index: int) -> tuple[str, Any]:
    base_line = lines[start_index]
    text_parts = [base_line["text"]]
    bbox = fitz.Rect(base_line["bbox"])
    last_rect = fitz.Rect(base_line["bbox"])
    previous_line = base_line
    continuation_count = 0

    for candidate in lines[start_index + 1 :]:
        candidate_rect = fitz.Rect(candidate["bbox"])
        if candidate_rect.y0 < bbox.y0 - CAPTION_ROW_TOLERANCE:
            continue
        if _caption_type(candidate["text"]) is not None:
            if abs(candidate_rect.y0 - bbox.y0) <= CAPTION_ROW_TOLERANCE:
                continue
            break
        if _is_same_row_caption_continuation(last_rect, candidate_rect, base_line) and _looks_like_caption_continuation_text(
            base_line,
            candidate,
            allow_code_only_fallback=continuation_count == 0,
        ):
            if continuation_count >= CAPTION_MAX_PARTS - 1:
                break
            text_parts.append(candidate["text"])
            bbox.include_rect(candidate_rect)
            last_rect = candidate_rect
            previous_line = candidate
            continuation_count += 1
            continue
        if continuation_count >= CAPTION_MAX_PARTS - 1:
            break
        if _is_next_row_caption_continuation(
            base_line,
            bbox,
            candidate,
            previous_line,
            continuation_count == 0,
        ):
            text_parts.append(candidate["text"])
            bbox.include_rect(candidate_rect)
            last_rect = candidate_rect
            previous_line = candidate
            continuation_count += 1
            continue
        if candidate_rect.y0 - bbox.y1 > CAPTION_CONTINUATION_GAP:
            break

    return _clean_line_text(" ".join(text_parts)), bbox


def _is_same_row_caption_continuation(previous_rect: Any, candidate_rect: Any, base_line: dict[str, Any]) -> bool:
    page_width = float(base_line.get("page_width") or 0)
    if (
        page_width > 0
        and CAPTION_CODE_ONLY_PATTERN.match(str(base_line.get("text") or "")) is None
        and _crosses_two_column_gutter(previous_rect, candidate_rect, page_width)
    ):
        return False
    return (
        abs(candidate_rect.y0 - previous_rect.y0) <= CAPTION_ROW_TOLERANCE
        and 0 <= candidate_rect.x0 - previous_rect.x1 <= CAPTION_SAME_ROW_GAP
    )


def _crosses_two_column_gutter(left: Any, right: Any, page_width: float) -> bool:
    mid_x = page_width / 2
    gutter = 10.0
    return (left.x1 <= mid_x - gutter and right.x0 >= mid_x + gutter) or (
        right.x1 <= mid_x - gutter and left.x0 >= mid_x + gutter
    )


def _is_next_row_caption_continuation(
    base_line: dict[str, Any],
    current_bbox: Any,
    candidate: dict[str, Any],
    previous_line: dict[str, Any],
    allow_code_only_fallback: bool,
) -> bool:
    candidate_rect = fitz.Rect(candidate["bbox"])
    previous_rect = fitz.Rect(previous_line["bbox"])
    if candidate_rect.y0 < previous_rect.y0 - CAPTION_ROW_TOLERANCE:
        return False
    if candidate_rect.y0 - previous_rect.y1 > CAPTION_CONTINUATION_GAP:
        return False

    base_rect = fitz.Rect(base_line["bbox"])
    base_center = (base_rect.x0 + base_rect.x1) / 2
    previous_center = (previous_rect.x0 + previous_rect.x1) / 2
    current_center = (current_bbox.x0 + current_bbox.x1) / 2
    candidate_center = (candidate_rect.x0 + candidate_rect.x1) / 2
    aligned = (
        _horizontal_overlap(base_rect, candidate_rect) > 0
        or _horizontal_overlap(previous_rect, candidate_rect) > 0
        or _horizontal_overlap(current_bbox, candidate_rect) > 0
        or abs(base_center - candidate_center) <= CAPTION_CENTER_TOLERANCE
        or abs(previous_center - candidate_center) <= CAPTION_CENTER_TOLERANCE
        or abs(current_center - candidate_center) <= CAPTION_CENTER_TOLERANCE
    )
    if not aligned:
        return False
    return _looks_like_caption_continuation_text(
        base_line,
        candidate,
        allow_code_only_fallback=allow_code_only_fallback,
    )


def _looks_like_caption_continuation_text(
    base_line: dict[str, Any],
    candidate: dict[str, Any],
    *,
    allow_code_only_fallback: bool = True,
) -> bool:
    text = str(candidate.get("text") or "").strip()
    if not text:
        return False
    if CONTINUED_PATTERN.match(text):
        return True
    font_size = float(candidate.get("font_size") or 0)
    base_font_size = float(base_line.get("font_size") or 0)
    if candidate.get("bold") and font_size <= base_font_size + 2:
        return True
    if _uppercase_ratio(text) >= 0.55 and font_size <= base_font_size + 2:
        return True
    return (
        allow_code_only_fallback
        and CAPTION_CODE_ONLY_PATTERN.match(str(base_line.get("text") or "")) is not None
        and font_size <= base_font_size + 2
    )


def _uppercase_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for char in letters if char.upper() == char) / len(letters)


def _is_continued_caption_suffix(line: dict[str, Any], next_line: dict[str, Any] | None) -> bool:
    if next_line is None or not CONTINUED_PATTERN.match(next_line["text"]):
        return False
    current = fitz.Rect(line["bbox"])
    candidate = fitz.Rect(next_line["bbox"])
    return (
        abs(candidate.y0 - current.y0) <= CAPTION_ROW_TOLERANCE
        and 0 <= candidate.x0 - current.x1 <= CAPTION_CONTINUED_GAP
    )


def _asset_clip_for_caption(page: Any, captions: list[AssetCaption], caption_index: int) -> Any | None:
    page_rect = fitz.Rect(page.rect)
    caption = captions[caption_index]
    caption_rect = fitz.Rect(caption.bbox)
    content_rect = _page_content_rect(page_rect)

    if _looks_like_landscape_asset_page(page_rect, caption_rect):
        return fitz.Rect(page_rect)

    column_rect = _caption_column_rect(page_rect, caption_rect)
    next_caption_y = _next_caption_y(captions, caption_index, column_rect)
    bottom_limit = next_caption_y - 8 if next_caption_y is not None else content_rect.y1

    if caption.asset_type == "figure":
        graphic_clip = _figure_clip_from_graphics(page, caption_rect, column_rect, bottom_limit)
        if graphic_clip is not None:
            graphic_clip.include_rect(caption_rect)
            return _clamp_rect(graphic_clip + (-8, -8, 8, 8), page_rect)

    table_clip = _table_clip_from_content(page, caption_rect, column_rect, bottom_limit)
    if table_clip is not None:
        return _clamp_rect(table_clip + (-8, -8, 8, 8), page_rect)

    fallback = fitz.Rect(column_rect.x0, caption_rect.y0, column_rect.x1, bottom_limit)
    return _clamp_rect(fallback, page_rect)


def _looks_like_landscape_asset_page(page_rect: Any, caption_rect: Any) -> bool:
    return page_rect.width > page_rect.height or caption_rect.height > caption_rect.width * 2


def _page_content_rect(page_rect: Any) -> Any:
    return fitz.Rect(
        BODY_SIDE_MARGIN,
        BODY_TOP_MARGIN,
        max(BODY_SIDE_MARGIN + 1, page_rect.width - BODY_SIDE_MARGIN),
        max(BODY_TOP_MARGIN + 1, page_rect.height - BODY_BOTTOM_MARGIN),
    )


def _caption_column_rect(page_rect: Any, caption_rect: Any) -> Any:
    content = _page_content_rect(page_rect)
    mid_x = page_rect.width / 2
    caption_center = (caption_rect.x0 + caption_rect.x1) / 2
    if abs(caption_center - mid_x) <= 45 or caption_rect.width >= page_rect.width * 0.42:
        return content
    if caption_center < mid_x:
        return fitz.Rect(content.x0, content.y0, mid_x - 10, content.y1)
    return fitz.Rect(mid_x + 10, content.y0, content.x1, content.y1)


def _next_caption_y(captions: list[AssetCaption], caption_index: int, column_rect: Any) -> float | None:
    current_y = fitz.Rect(captions[caption_index].bbox).y0
    candidates: list[float] = []
    for index, caption in enumerate(captions):
        if index == caption_index:
            continue
        rect = fitz.Rect(caption.bbox)
        if rect.y0 <= current_y:
            continue
        if _horizontal_overlap(rect, column_rect) <= 0:
            continue
        candidates.append(rect.y0)
    return min(candidates) if candidates else None


def _table_clip_from_content(page: Any, caption_rect: Any, column_rect: Any, bottom_limit: float) -> Any | None:
    search_rect = fitz.Rect(column_rect.x0, caption_rect.y0, column_rect.x1, bottom_limit)
    drawing_rects = _related_table_drawing_rects(page, caption_rect, column_rect, bottom_limit)
    if drawing_rects:
        max_drawing_y = min(max(rect.y1 for rect in drawing_rects), bottom_limit)
        text_items = [
            item
            for item in _page_text_items(page)
            if item["rect"].y1 >= caption_rect.y0
            and item["rect"].y0 <= bottom_limit
            and _horizontal_overlap(item["rect"], column_rect) > 1
        ]
        text_rects = [
            item["rect"]
            for item in text_items
            if item["rect"].y0 <= max_drawing_y + 3
        ]
        text_rects.extend(_table_note_rects_after_rule(text_items, max_drawing_y=max_drawing_y))
        return _union_rects([caption_rect, *drawing_rects, *text_rects])

    text_clip = _table_clip_from_text_rows(page, caption_rect, column_rect, search_rect)
    if text_clip is not None:
        return text_clip
    return None


def _related_table_drawing_rects(page: Any, caption_rect: Any, column_rect: Any, bottom_limit: float) -> list[Any]:
    candidates = sorted(
        (
            rect
            for rect in _page_drawing_rects(page)
            if rect.y1 >= caption_rect.y1 and rect.y0 <= bottom_limit and _horizontal_overlap(rect, column_rect) > 1
        ),
        key=lambda rect: (max(0.0, rect.y0 - caption_rect.y1), rect.y0, rect.x0),
    )
    if not candidates:
        return []

    seed = candidates[0]
    if seed.y0 - caption_rect.y1 > TABLE_DRAWING_START_GAP:
        return []

    related = [fitz.Rect(seed)]
    union = fitz.Rect(seed)
    for rect in sorted(candidates[1:], key=lambda item: (item.y0, item.x0)):
        if rect.y0 - union.y1 > TABLE_DRAWING_CHAIN_GAP:
            break
        if _horizontal_overlap(rect, union) <= 0 and _horizontal_overlap(rect, column_rect) < min(rect.width, column_rect.width) * 0.25:
            continue
        related.append(rect)
        union.include_rect(rect)
    return related


def _table_clip_from_text_rows(page: Any, caption_rect: Any, column_rect: Any, bounds: Any) -> Any | None:
    items = sorted(
        (
            item
            for item in _page_text_items(page)
            if item["rect"].y1 >= caption_rect.y0
            and item["rect"].y0 <= bounds.y1
            and _horizontal_overlap(item["rect"], column_rect) > 1
        ),
        key=lambda item: (item["rect"].y0, item["rect"].x0),
    )
    if not items:
        return None

    output = fitz.Rect(caption_rect)
    previous = caption_rect
    consumed = False
    for item in items:
        rect = item["rect"]
        if rect.y1 <= caption_rect.y1:
            continue
        if rect.y0 - previous.y1 > TABLE_TEXT_CHAIN_GAP:
            break
        if consumed and _looks_like_body_text_after_table(item):
            break
        if not _looks_like_table_text_item(item):
            if consumed:
                break
            continue
        output.include_rect(rect)
        previous = rect
        consumed = True

    return output if consumed else None


def _figure_clip_from_graphics(page: Any, caption_rect: Any, column_rect: Any, bottom_limit: float) -> Any | None:
    candidates = [
        rect
        for rect in [*_page_image_rects(page), *_page_drawing_rects(page)]
        if _horizontal_overlap(rect, column_rect) > 1
        and rect.y1 >= column_rect.y0
        and rect.y0 <= bottom_limit
        and abs(_vertical_gap(rect, caption_rect)) <= 280
    ]
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda rect: abs(_vertical_gap(rect, caption_rect)))
    seed = candidates[0]
    related = [
        rect
        for rect in candidates
        if _rects_near_vertically(seed, rect, max_gap=60) or _rects_near_vertically(caption_rect, rect, max_gap=120)
    ]
    return _union_rects(related)


def _page_drawing_rects(page: Any) -> list[Any]:
    rects: list[Any] = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None:
            continue
        item = fitz.Rect(rect)
        if item.width <= 0 and item.height <= 0:
            continue
        if item.width < 3 and item.height < 3:
            continue
        rects.append(item)
    return rects


def _page_image_rects(page: Any) -> list[Any]:
    rects: list[Any] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 1:
            continue
        bbox = block.get("bbox")
        if bbox:
            rects.append(fitz.Rect(bbox))
    return rects


def _page_text_rects(page: Any) -> list[Any]:
    return [item["rect"] for item in _page_text_items(page)]


def _page_text_items(page: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for block in page.get_text("dict", sort=True).get("blocks", []):
        if block.get("type") != 0:
            continue
        for raw_line in block.get("lines", []):
            spans = raw_line.get("spans", [])
            text = _clean_line_text(" ".join(str(span.get("text") or "") for span in spans))
            boxes = [span.get("bbox") for span in spans if _clean_line_text(str(span.get("text") or ""))]
            boxes = [box for box in boxes if box]
            if boxes:
                sizes = [float(span.get("size") or 0) for span in spans if float(span.get("size") or 0) > 0]
                items.append(
                    {
                        "text": text,
                        "rect": _union_rects(fitz.Rect(box) for box in boxes),
                        "font_size": max(sizes) if sizes else 0.0,
                    }
                )
    return items


def _table_note_rects_after_rule(text_items: list[dict[str, Any]], *, max_drawing_y: float) -> list[Any]:
    note_rects: list[Any] = []
    current_y = max_drawing_y
    for item in sorted(text_items, key=lambda value: (value["rect"].y0, value["rect"].x0)):
        rect = item["rect"]
        if rect.y0 <= max_drawing_y + 3:
            continue
        if rect.y0 - current_y > TABLE_NOTE_MAX_GAP:
            break
        if _looks_like_table_note_item(item):
            note_rects.append(rect)
            current_y = max(current_y, rect.y1)
            continue
        break
    return note_rects


def _looks_like_table_note_item(item: dict[str, Any]) -> bool:
    text = str(item.get("text") or "").strip()
    font_size = float(item.get("font_size") or 0)
    if font_size <= TABLE_NOTE_MAX_FONT_SIZE:
        return True
    return bool(re.match(r"^(?:N\s*OTE|NOTE|[A-Z](?:\s|$))", text, re.IGNORECASE)) and len(text) <= 220


def _looks_like_table_text_item(item: dict[str, Any]) -> bool:
    text = str(item.get("text") or "").strip()
    font_size = float(item.get("font_size") or 0)
    if not text:
        return False
    if font_size <= 8.6:
        return True
    if len(text) <= 80 and _uppercase_ratio(text) >= 0.35:
        return True
    if re.search(r"(?:\d|%|\[[^\]]+\]|\.{2,}|—|–)", text) and len(text) <= 120:
        return True
    return False


def _looks_like_body_text_after_table(item: dict[str, Any]) -> bool:
    text = str(item.get("text") or "").strip()
    font_size = float(item.get("font_size") or 0)
    if font_size >= 9.0 and len(text) >= 45:
        return True
    return bool(re.match(r"^(?:\d{1,2}|S\d{1,3})(?:\.\d+)*\s+", text)) and font_size >= 8.8


def _content_union_rect(page: Any, *, fallback: Any) -> Any:
    rects = [*_page_text_rects(page), *_page_image_rects(page), *_page_drawing_rects(page)]
    if not rects:
        return fitz.Rect(fallback)
    content = _union_rects(rects)
    content.intersect(fallback)
    return content if not content.is_empty else fitz.Rect(fallback)


def _union_until_large_gap(rects: list[Any], bounds: Any) -> Any:
    filtered = sorted(
        (fitz.Rect(rect) for rect in rects if fitz.Rect(rect).intersects(bounds)),
        key=lambda rect: (rect.y0, rect.x0),
    )
    if not filtered:
        return fitz.Rect(bounds)
    output = fitz.Rect(filtered[0])
    previous = filtered[0]
    for rect in filtered[1:]:
        if rect.y0 - previous.y1 > 95:
            break
        output.include_rect(rect)
        previous = rect
    return output


def _union_rects(rects: Any) -> Any:
    iterator = iter(rects)
    first = fitz.Rect(next(iterator))
    for rect in iterator:
        first.include_rect(fitz.Rect(rect))
    return first


def _horizontal_overlap(left: Any, right: Any) -> float:
    return max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))


def _vertical_gap(left: Any, right: Any) -> float:
    if left.y1 < right.y0:
        return right.y0 - left.y1
    if right.y1 < left.y0:
        return left.y0 - right.y1
    return 0.0


def _rects_near_vertically(left: Any, right: Any, *, max_gap: float) -> bool:
    return _vertical_gap(left, right) <= max_gap


def _clamp_rect(rect: Any, page_rect: Any) -> Any:
    output = fitz.Rect(rect)
    output.intersect(page_rect)
    return output


def _rect_tuple(rect: Any) -> tuple[float, float, float, float]:
    item = fitz.Rect(rect)
    return (round(item.x0, 2), round(item.y0, 2), round(item.x1, 2), round(item.y1, 2))


def _title_page_candidate(
    page: Any,
    *,
    page_index: int,
    title_page_max_chars: int,
) -> StandardTitlePage | None:
    lines = _extract_title_lines(page, page_index=page_index)
    page_height = float(page.rect.height)
    page_text = "\n".join(line.text for line in lines)
    page_text_chars = _text_length(page_text)
    if page_text_chars > title_page_max_chars:
        return None

    standard_code_count = sum(1 for line in lines if line.is_standard_code)
    if standard_code_count not in {1, 2}:
        return None

    standard_lines = [
        line
        for line in lines
        if line.is_standard_code and _is_body_title_area(line, page_height)
    ]
    if not standard_lines:
        return None

    standard_font_size = max(line.font_size for line in standard_lines)
    first_standard_y0 = min(line.y0 for line in standard_lines)
    title_lines = [
        line
        for line in lines
        if line in standard_lines
        or (line.y0 < first_standard_y0 and line.font_size > standard_font_size + FONT_SIZE_TOLERANCE)
    ]
    if not title_lines:
        return None

    title_text = " ".join(line.text for line in sorted(title_lines, key=lambda item: item.order)).strip()
    standard_code = _standard_code_from_text("\n".join(line.text for line in standard_lines))
    if not standard_code:
        return None

    return StandardTitlePage(
        page_number=page_index + 1,
        page_index=page_index,
        title=title_text,
        standard_code=standard_code,
        page_text_chars=page_text_chars,
        standard_code_count=standard_code_count,
        standard_font_size=standard_font_size,
        title_lines=title_lines,
    )


def _extract_title_lines(page: Any, *, page_index: int) -> list[StandardTitleLine]:
    output: list[StandardTitleLine] = []
    blocks = page.get_text("dict", sort=True).get("blocks", [])
    order = 0
    for block in blocks:
        if block.get("type") != 0:
            continue
        for raw_line in block.get("lines", []):
            text_parts: list[str] = []
            sizes: list[float] = []
            y0_values: list[float] = []
            for span in raw_line.get("spans", []):
                text = _clean_line_text(str(span.get("text") or ""))
                if not text:
                    continue
                text_parts.append(text)
                size = float(span.get("size") or 0)
                if size > 0:
                    sizes.append(size)
                bbox = span.get("bbox") or (0, 0, 0, 0)
                y0_values.append(float(bbox[1]))
            text = _clean_line_text(" ".join(text_parts))
            if not text:
                continue
            output.append(
                StandardTitleLine(
                    text=text,
                    font_size=max(sizes) if sizes else 0.0,
                    y0=min(y0_values) if y0_values else 0.0,
                    order=(page_index * 10000) + order,
                    is_standard_code=_is_standard_code_line(text),
                )
            )
            order += 1
    return output


def _is_standard_code_line(text: str) -> bool:
    return bool(STANDARD_CODE_LINE_PATTERN.match(_clean_line_text(text)))


def _is_body_title_area(line: StandardTitleLine, page_height: float) -> bool:
    if page_height <= 0:
        return True
    return TITLE_PAGE_CODE_MIN_Y_RATIO * page_height <= line.y0 <= TITLE_PAGE_CODE_MAX_Y_RATIO * page_height


def _section_ranges(
    title_pages: list[StandardTitlePage],
    content_start_index: int,
    content_end_index: int,
) -> list[tuple[StandardTitlePage | None, int, int]]:
    if not title_pages:
        return [(None, content_start_index, content_end_index)]

    ranges: list[tuple[StandardTitlePage | None, int, int]] = []
    for index, title_page in enumerate(title_pages):
        start_index = max(title_page.page_index, content_start_index)
        next_start = title_pages[index + 1].page_index if index + 1 < len(title_pages) else content_end_index + 1
        end_index = min(next_start - 1, content_end_index)
        if start_index <= end_index:
            ranges.append((title_page, start_index, end_index))
    return ranges


def _build_manifest_items(sections: list[StandardPdfSection], manifest_path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = [
        {
            "type": "paragraph",
            "style": HEADING_STYLE,
            "text": "ASME standard PDF split manifest",
            "source": "standard_pdf_parser:manifest",
        },
        {
            "type": "paragraph",
            "style": BODY_STYLE,
            "text": f"Manifest: {manifest_path}",
            "source": "standard_pdf_parser:manifest",
        },
    ]
    for section in sections:
        items.append(
            {
                "type": "paragraph",
                "style": BODY_STYLE,
                "text": (
                    f"{section.index}. {section.title} "
                    f"(pages {section.start_page}-{section.end_page}) -> {section.output_path}"
                ),
                "source": f"standard_pdf_parser:section:{section.index}",
            }
        )
    return items


def _clear_existing_standard_outputs(output_dir: Path) -> None:
    for child in output_dir.iterdir() if output_dir.exists() else []:
        if child.is_dir() and ((child / "pdf").exists() or (child / "img").exists() or (child / "text_pdf").exists()):
            shutil.rmtree(child)
    for legacy_name in ("pdf_sections", "assets"):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists() and legacy_path.is_dir():
            shutil.rmtree(legacy_path)


def _clear_existing_section_images(source_path: Path, sections: list[StandardPdfSection]) -> None:
    for section in sections:
        image_dir = _section_image_dir(source_path, section)
        if image_dir.exists():
            for path in image_dir.glob("*.png"):
                if path.is_file():
                    path.unlink()


def _clear_existing_section_text_pdfs(source_path: Path, sections: list[StandardPdfSection]) -> None:
    for section in sections:
        text_pdf_dir = _section_text_pdf_dir(source_path, section)
        if text_pdf_dir.exists():
            for path in text_pdf_dir.glob("*.pdf"):
                if path.is_file():
                    path.unlink()


def _clear_legacy_asset_dir(output_dir: Path) -> None:
    legacy_path = output_dir / "assets"
    if legacy_path.exists() and legacy_path.is_dir():
        shutil.rmtree(legacy_path)


def _section_base_name(title: str, standard_code: str) -> str:
    name_parts: list[str] = []
    if standard_code:
        name_parts.append(standard_code)
    name_parts.append(title)
    filename = " - ".join(_sanitize_filename_part(part) for part in name_parts if part)
    return filename[:180].rstrip() or "section"


def _standard_split_object_prefix(object_name: str) -> str:
    source_path = Path(str(object_name).replace("\\", "/"))
    split_dir_name = f"{source_path.stem}(切分版)"
    parent = source_path.parent.as_posix()
    return f"{parent}/{split_dir_name}" if parent and parent != "." else split_dir_name


def _normalize_split_prefix(prefix: str | Path) -> str:
    return str(prefix).replace("\\", "/").strip("/")


def _standard_source_reference(source_ref: str | Path, *, bucket: str | None = None):
    source_text = str(source_ref)
    source_path = Path(source_text)
    if source_path.exists():
        object_name = raw_document_object_name_for_file(source_path)
        return parse_raw_document_reference(object_name, bucket=bucket)
    return parse_raw_document_reference(source_ref, bucket=bucket)


def _sanitize_filename_part(value: str) -> str:
    cleaned = FILENAME_UNSAFE_PATTERN.sub("_", str(value or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._")
    return cleaned or "untitled"


def _unique_name(name: str, used_names: set[str]) -> str:
    candidate = name
    suffix = 2
    while candidate.lower() in used_names:
        suffix_text = f" ({suffix})"
        candidate = f"{name[: 180 - len(suffix_text)]}{suffix_text}"
        suffix += 1
    used_names.add(candidate.lower())
    return candidate


def _unique_filename(filename: str, used_names: set[str]) -> str:
    candidate = filename
    suffix = 2
    while candidate.lower() in used_names:
        path = Path(filename)
        stem = path.stem
        extension = path.suffix or ".png"
        suffix_text = f" ({suffix})"
        candidate = f"{stem[: 180 - len(suffix_text) - len(extension)]}{suffix_text}{extension}"
        suffix += 1
    used_names.add(candidate.lower())
    return candidate


def _section_image_dir(source_path: Path, section: StandardPdfSection) -> Path:
    if section.section_dir:
        return Path(section.section_dir) / "img"
    section_dir = processing_document_dir(source_path) / _section_base_name(section.title, section.standard_code)
    return section_dir / "img"


def _section_text_pdf_dir(source_path: Path, section: StandardPdfSection) -> Path:
    if section.section_dir:
        return Path(section.section_dir) / "text_pdf"
    section_dir = processing_document_dir(source_path) / _section_base_name(section.title, section.standard_code)
    return section_dir / "text_pdf"


def _asset_filename(caption: AssetCaption) -> str:
    caption_key = _sanitize_filename_part(caption.text)
    return f"{caption.asset_type} - {caption_key[:150]}.png"


def _title_page_to_dict(title_page: StandardTitlePage) -> dict[str, Any]:
    payload = asdict(title_page)
    payload["title_lines"] = [asdict(line) for line in title_page.title_lines]
    return payload


def _standard_code_from_text(text: str) -> str:
    match = STANDARD_CODE_PATTERN.search(text)
    if not match:
        return ""
    return _normalize_standard_code(match.group(0))


def _normalize_standard_code(text: str) -> str:
    value = _clean_line_text(text).upper()
    value = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", value)
    value = re.sub(r"\s*/\s*", "/", value)
    value = re.sub(r"\s*-\s*", "-", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _is_bold_font(font_name: str, flags: int) -> bool:
    normalized = font_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    positive_tokens = ("bold", "black", "heavy", "semibold", "demibold", "extrabold", "ultrabold")
    negative_tokens = ("regular", "normal", "book", "light", "thin", "medium", "roman")
    if any(token in normalized for token in positive_tokens):
        return True
    if any(token in normalized for token in negative_tokens):
        return False
    return bool(flags & 16)


def _clean_line_text(text: str) -> str:
    value = str(text or "").replace("\u00a0", " ").replace("\u3000", " ")
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


def _text_length(text: str) -> int:
    return len(re.sub(r"\s+", "", str(text or "")))


def write_items_to_txt(items: list[dict[str, Any]], source_file: str | Path) -> Path:
    source_path = Path(source_file)
    txt_dir = processing_subdir(source_path, "txt")
    txt_dir.mkdir(parents=True, exist_ok=True)

    output_path = txt_dir / f"{source_path.stem}.txt"
    output_path.write_text(format_extracted_items(items), encoding="utf-8")
    return output_path


def write_section_items_to_txt(items: list[dict[str, Any]], section: StandardPdfSection) -> Path:
    section_path = Path(section.output_path)
    section_dir = Path(section.section_dir) if section.section_dir else section_path.parent.parent
    txt_dir = section_dir / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    output_path = txt_dir / f"{section_path.stem}.txt"
    output_path.write_text(format_extracted_items(items), encoding="utf-8")
    return output_path


def write_section_referenced_documents_to_json(items: list[dict[str, Any]], section: StandardPdfSection) -> Path:
    section_path = Path(section.output_path)
    section_dir = Path(section.section_dir) if section.section_dir else section_path.parent.parent
    json_dir = section_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    output_path = json_dir / "referenced_documents.json"
    text = "\n".join(str(item.get("text") or "").strip() for item in items if str(item.get("text") or "").strip())
    payload = {
        "section_index": section.index,
        "title": section.title,
        "standard_code": section.standard_code,
        "start_page": section.start_page,
        "end_page": section.end_page,
        "source_pdf": section.source_uri or section.output_path,
        "local_pdf": section.output_path,
        "text_pdf": section.text_pdf_path,
        "text": text,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _ensure_dependencies() -> None:
    _ensure_fitz()


def _ensure_fitz() -> None:
    if fitz is None:
        raise ModuleNotFoundError("pymupdf is required to split standard PDF documents")


def _log(message: str) -> None:
    print(f"[standard_pdf_parser] {message}", flush=True)
