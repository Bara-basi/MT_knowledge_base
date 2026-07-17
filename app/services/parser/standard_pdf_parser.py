from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.parser.standard_pdf_splitter import (
    DEFAULT_TITLE_PAGE_MAX_CHARS,
    StandardAsset,
    StandardPdfSection,
    extract_and_write_standard_section_texts,
    extract_standard_assets_from_sections,
    extract_standard_text_from_sections,
    find_standard_title_pages,
    load_standard_assets_manifest,
    split_standard_pdf,
    split_standard_pdf_document,
    write_masked_text_pdfs,
)


def parse_standard_pdf_document(
    file_path: str | Path,
    *,
    image_analysis_workers: int = 3,
    title_page_max_chars: int = DEFAULT_TITLE_PAGE_MAX_CHARS,
    source_reference: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Parse a standard PDF by first splitting it into source sections."""

    sections = split_standard_pdf_document(
        file_path,
        image_analysis_workers=image_analysis_workers,
        title_page_max_chars=title_page_max_chars,
        source_reference=source_reference,
    )
    return parse_standard_pdf_sections(sections)


def parse_standard_pdf_sections(
    sections: list[StandardPdfSection],
) -> list[dict[str, Any]]:
    """Parse text from already split per-standard PDFs and write section txt files."""
    items, _txt_paths = extract_and_write_standard_section_texts(sections)
    return items
