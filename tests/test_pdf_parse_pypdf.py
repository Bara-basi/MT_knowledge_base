from __future__ import annotations

from pathlib import Path

from pdf_parse_compare_common import build_page, run_parser


def parse_pdf(pdf_path: Path) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path), strict=False)
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(
            build_page(
                index,
                text,
                rotation=page.get("/Rotate", 0),
                mediabox=[float(value) for value in page.mediabox],
            )
        )
    return pages


if __name__ == "__main__":
    run_parser("pypdf", parse_pdf)
