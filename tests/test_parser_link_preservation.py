from __future__ import annotations

from openpyxl import Workbook

from app.services.chunking.splitter import split_items
from app.services.parser.excel_parser import _row_links
from app.services.parser.link_parser import enrich_links
from app.services.parser.word_parser import _clean_text


def test_attachment_placeholders_are_not_removed_from_word_or_link_cleaning() -> None:
    text = "See embedded file [deck.pptx] and [sheet.xlsx]"

    assert _clean_text(text) == text
    assert enrich_links([{"type": "paragraph", "style": "Body", "text": text}], llm_client=None) == [
        {"type": "paragraph", "style": "Body", "text": text}
    ]


def test_table_links_survive_chunk_metadata_filtering() -> None:
    chunks = split_items(
        [
            {
                "type": "table",
                "style": "Table",
                "text": '[{"name":"Logo","owner":"Sales"}]',
                "links": {
                    "Logo": "https://example.com/logo",
                    "Brochure": "https://example.com/brochure",
                },
            }
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].content.count("<a index=") == 2
    assert chunks[0].metadata["links"] == [
        {"index": 1, "link_name": "Logo", "link_path": "https://example.com/logo"},
        {"index": 2, "link_name": "Brochure", "link_path": "https://example.com/brochure"},
    ]


def test_excel_row_links_extracts_cell_hyperlinks() -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(["name", "asset"])
    worksheet.append(["Coil", "Open asset"])
    worksheet["B2"].hyperlink = "https://example.com/asset"

    assert _row_links(worksheet, 2, ["name", "asset"]) == {
        "Open asset": "https://example.com/asset"
    }
