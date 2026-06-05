from __future__ import annotations

from pathlib import Path

from pdf_parse_compare_common import build_page, run_parser


def parse_pdf(pdf_path: Path) -> list[dict]:
    import fitz

    pages = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text") or ""
            blocks = page.get_text("blocks") or []
            pages.append(
                build_page(
                    index,
                    text,
                    width=page.rect.width,
                    height=page.rect.height,
                    block_count=len(blocks),
                    blocks=[
                        {
                            "bbox": [block[0], block[1], block[2], block[3]],
                            "text": block[4],
                            "block_no": block[5],
                            "block_type": block[6],
                        }
                        for block in blocks
                    ],
                )
            )
    return pages


if __name__ == "__main__":
    run_parser("pymupdf", parse_pdf)
