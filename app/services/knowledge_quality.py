"""Conservative admission rules for automatically ingested knowledge files."""

from __future__ import annotations

from pathlib import Path


SUPPORTED_OFFICE_SUFFIXES = frozenset({".docx", ".xlsx", ".pptx"})
MIN_PDF_NATIVE_TEXT_CHARS_PER_PAGE = 80
MIN_PDF_NATIVE_TEXT_CHARS = 300


def is_acceptable_knowledge_file(path: str | Path) -> tuple[bool, str]:
    """Accept stable Office formats and text PDFs; reject scans and everything else."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in SUPPORTED_OFFICE_SUFFIXES:
        return True, "accepted_office"
    if suffix != ".pdf":
        return False, f"unsupported_format:{suffix or 'none'}"
    try:
        import fitz
    except ImportError:
        return False, "pdf_text_inspector_unavailable"
    try:
        with fitz.open(file_path) as pdf:
            page_count = len(pdf)
            if page_count < 1:
                return False, "empty_pdf"
            native_text_chars = sum(len(page.get_text("text").strip()) for page in pdf)
    except Exception:
        return False, "unreadable_pdf"
    minimum = max(MIN_PDF_NATIVE_TEXT_CHARS, page_count * MIN_PDF_NATIVE_TEXT_CHARS_PER_PAGE)
    if native_text_chars < minimum:
        return False, "scanned_or_low_text_pdf"
    return True, "accepted_native_text_pdf"
