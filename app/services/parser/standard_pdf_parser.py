from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.parser.standard_pdf_splitter import (
    DEFAULT_TITLE_PAGE_MAX_CHARS,
    extract_and_write_standard_section_texts,
    split_standard_pdf_document,
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
    items, _txt_paths = extract_and_write_standard_section_texts(sections)
    return items
