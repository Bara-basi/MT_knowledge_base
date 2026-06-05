from __future__ import annotations

from pathlib import Path

from pdf_parse_compare_common import build_page, run_parser


def parse_pdf(pdf_path: Path) -> list[dict]:
    import pdfplumber

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            pages.append(
                build_page(
                    index,
                    text,
                    width=page.width,
                    height=page.height,
                    char_count=len(page.chars),
                    table_count=len(tables),
                    tables=tables,
                )
            )
    return pages


if __name__ == "__main__":
    run_parser("pdfplumber", parse_pdf)
