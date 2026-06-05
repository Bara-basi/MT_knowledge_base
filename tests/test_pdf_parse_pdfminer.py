from __future__ import annotations

from pathlib import Path

from pdf_parse_compare_common import build_page, run_parser


def parse_pdf(pdf_path: Path) -> list[dict]:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    pages = []
    for index, layout in enumerate(extract_pages(str(pdf_path)), start=1):
        text_blocks = []
        for element in layout:
            if isinstance(element, LTTextContainer):
                text_blocks.append(
                    {
                        "bbox": list(element.bbox),
                        "text": element.get_text(),
                    }
                )

        text = "\n".join(block["text"].rstrip() for block in text_blocks)
        pages.append(
            build_page(
                index,
                text,
                width=layout.width,
                height=layout.height,
                text_block_count=len(text_blocks),
                text_blocks=text_blocks,
            )
        )
    return pages


if __name__ == "__main__":
    run_parser("pdfminer", parse_pdf)
