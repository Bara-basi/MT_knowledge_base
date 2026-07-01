from __future__ import annotations

from pathlib import Path

import fitz

from app.services.parser.standard_pdf_parser import (
    StandardAsset,
    StandardPdfSection,
    extract_standard_assets_from_sections,
    extract_standard_text_from_sections,
    find_standard_title_pages,
    load_standard_assets_manifest,
    split_standard_pdf,
    write_masked_text_pdfs,
)
from app.services.parser.paths import processing_document_dir


def _insert_centered_text(page, text: str, y: int, *, fontsize: int) -> None:
    page_width = page.rect.width
    text_width = fitz.get_text_length(text, fontsize=fontsize)
    page.insert_text(((page_width - text_width) / 2, y), text, fontsize=fontsize)


def _write_title_page_pdf(tmp_path: Path, filename: str, lines: list[tuple[str, int, int]]) -> Path:
    source = tmp_path / filename
    document = fitz.open()
    document.new_page().insert_text((72, 72), "cover")
    page = document.new_page()
    for text, y, fontsize in lines:
        page.insert_text((72, y), text, fontsize=fontsize)
    document.new_page().insert_text((72, 72), "body")
    document.new_page().insert_text((72, 72), "back cover")
    document.save(source)
    document.close()
    return source


def test_standard_pdf_title_page_detection_uses_sa_code_and_larger_title_font(tmp_path: Path) -> None:
    source = tmp_path / "ASME-demo.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "cover")
    title_page = document.new_page()
    _insert_centered_text(title_page, "SPECIFICATION FOR GENERAL REQUIREMENTS FOR", 150, fontsize=18)
    _insert_centered_text(title_page, "STEEL PLATES FOR PRESSURE VESSELS", 180, fontsize=18)
    _insert_centered_text(title_page, "SA-20/SA-20M", 215, fontsize=12)
    document.new_page().insert_text((72, 72), "body")
    document.new_page().insert_text((72, 72), "back cover")
    document.save(source)
    document.close()

    title_pages = find_standard_title_pages(source)

    assert len(title_pages) == 1
    assert title_pages[0].page_number == 2
    assert title_pages[0].standard_code == "SA-20/SA-20M"
    assert title_pages[0].standard_code_count == 1
    assert title_pages[0].title == (
        "SPECIFICATION FOR GENERAL REQUIREMENTS FOR "
        "STEEL PLATES FOR PRESSURE VESSELS SA-20/SA-20M"
    )


def test_standard_pdf_split_skips_covers_and_puts_front_matter_into_first_section(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "ASME-demo.pdf"
    document = fitz.open()
    for text in [
        "cover",
        "front matter",
        None,
        "section body",
        None,
        "second body",
        "back cover",
    ]:
        page = document.new_page()
        if text is None:
            continue
        page.insert_text((72, 72), text, fontsize=12)
    first_title = document[2]
    first_title.insert_text((72, 246), "SPECIFICATION FOR GENERAL REQUIREMENTS FOR", fontsize=17)
    first_title.insert_text((72, 265), "STEEL PLATES FOR PRESSURE VESSELS", fontsize=17)
    first_title.insert_text((72, 338), "SA-20/SA-20M", fontsize=14)
    second_title = document[4]
    second_title.insert_text((72, 246), "SPECIFICATION FOR NICKEL ALLOY", fontsize=17)
    second_title.insert_text((72, 338), "SA-240", fontsize=14)
    document.save(source)
    document.close()
    monkeypatch.chdir(tmp_path)
    stale_pdf = tmp_path / "data" / "processing" / source.stem / "pdf_sections" / "stale.pdf"
    stale_pdf.parent.mkdir(parents=True, exist_ok=True)
    stale_pdf.write_bytes(b"old")

    title_pages = find_standard_title_pages(source, title_page_max_chars=160)
    sections = split_standard_pdf(source, title_pages)

    assert [(section.start_page, section.end_page) for section in sections] == [(2, 4), (5, 6)]
    assert not stale_pdf.exists()
    assert Path(sections[0].output_path).parent.name == "pdf"
    with fitz.open(sections[0].output_path) as first_section:
        assert len(first_section) == 3
    with fitz.open(sections[1].output_path) as second_section:
        assert len(second_section) == 2


def test_standard_pdf_title_page_detection_ignores_header_only_standard_codes(tmp_path: Path) -> None:
    source = tmp_path / "ASME-demo.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "cover")
    page = document.new_page()
    page.insert_text((72, 40), "ASME BPVC.II.A-2023", fontsize=8)
    page.insert_text((420, 40), "SA-53/SA-53M", fontsize=8)
    page.insert_text((72, 90), "1. Scope", fontsize=10)
    page.insert_text((72, 745), "158", fontsize=9)
    document.new_page().insert_text((72, 72), "back cover")
    document.save(source)
    document.close()

    assert find_standard_title_pages(source) == []


def test_standard_pdf_title_page_detection_ignores_cited_standard_codes(tmp_path: Path) -> None:
    source = _write_title_page_pdf(
        tmp_path,
        "ASME-demo.pdf",
        [
            ("SA-747/SA-747M", 40, 8),
            ("SPECIFICATION FOR STEEL CASTINGS, STAINLESS,", 246, 17),
            ("SA-747/SA-747M", 338, 14),
            ("Supplementary Requirement S15 of SA-781/SA-781M applies.", 416, 8),
        ],
    )

    title_pages = find_standard_title_pages(source)

    assert len(title_pages) == 1
    assert title_pages[0].standard_code == "SA-747/SA-747M"


def test_standard_pdf_title_page_detection_supports_sf_and_sb_en_codes(tmp_path: Path) -> None:
    sf_source = _write_title_page_pdf(
        tmp_path,
        "ASME-sf-demo.pdf",
        [
            ("SF-568M", 40, 8),
            ("SPECIFICATION FOR CARBON AND ALLOY STEEL", 246, 17),
            ("SF-568M", 338, 14),
        ],
    )
    sb_en_source = _write_title_page_pdf(
        tmp_path,
        "ASME-sb-en-demo.pdf",
        [
            ("SB/EN 1706", 40, 8),
            ("SPECIFICATION FOR ALUMINUM AND ALUMINUM ALLOYS", 246, 17),
            ("SB/EN 1706", 338, 14),
        ],
    )

    sf_titles = find_standard_title_pages(sf_source)
    sb_en_titles = find_standard_title_pages(sb_en_source)

    assert [title.standard_code for title in sf_titles] == ["SF-568M"]
    assert [title.standard_code for title in sb_en_titles] == ["SB/EN 1706"]


def test_standard_pdf_asset_extraction_renders_table_and_figure_crops(tmp_path: Path, monkeypatch) -> None:
    section_pdf = tmp_path / "section.pdf"
    document = fitz.open()
    page = document.new_page()
    _insert_centered_text(page, "TABLE 1", 80, fontsize=8)
    _insert_centered_text(page, "Chemical Requirements", 92, fontsize=8)
    page.draw_line((90, 105), (520, 105))
    page.insert_text((100, 125), "Element", fontsize=8)
    page.insert_text((300, 125), "Composition", fontsize=8)
    page.draw_line((90, 145), (520, 145))
    page.insert_text((100, 165), "Carbon", fontsize=8)
    page.insert_text((300, 165), "0.10 max", fontsize=8)
    page.draw_line((90, 185), (520, 185))
    figure_page = document.new_page()
    figure_page.draw_rect(fitz.Rect(160, 90, 450, 220))
    figure_page.insert_text((150, 250), "FIG. 1", fontsize=8)
    figure_page.insert_text((310, 250), "Test Specimen", fontsize=8)
    document.save(section_pdf)
    document.close()
    monkeypatch.chdir(tmp_path)
    stale_asset = tmp_path / "data" / "processing" / "ASME-demo" / "assets" / "stale.png"
    stale_asset.parent.mkdir(parents=True, exist_ok=True)
    stale_asset.write_bytes(b"old")

    manifest_path = extract_standard_assets_from_sections(
        tmp_path / "ASME-demo.pdf",
        [
            StandardPdfSection(
                index=1,
                title="Demo",
                standard_code="SA-1",
                start_page=10,
                end_page=11,
                output_path=str(section_pdf),
            )
        ],
    )

    assert manifest_path.exists()
    assert not stale_asset.exists()
    asset_files = sorted(manifest_path.parent.glob("*/img/*.png"))
    assert len(asset_files) == 2
    assert all(not path.name.startswith("000") for path in asset_files)
    assert all(path.parent.name == "img" for path in asset_files)
    with fitz.open(asset_files[0]) as rendered_table:
        assert len(rendered_table) == 1
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "TABLE 1 Chemical Requirements" in manifest_text
    assert "FIG. 1 Test Specimen" in manifest_text


def test_standard_pdf_asset_extraction_keeps_multiline_uppercase_caption(tmp_path: Path, monkeypatch) -> None:
    section_pdf = tmp_path / "section.pdf"
    document = fitz.open()
    page = document.new_page()
    _insert_centered_text(page, "TABLE 1", 80, fontsize=8)
    _insert_centered_text(page, "CHEMICAL REQUIREMENTS FOR", 92, fontsize=8)
    _insert_centered_text(page, "NICKEL AND NICKEL ALLOY", 104, fontsize=8)
    page.draw_line((90, 120), (520, 120))
    page.insert_text((100, 138), "Element", fontsize=8)
    page.insert_text((300, 138), "Composition", fontsize=8)
    page.draw_line((90, 156), (520, 156))
    page.insert_text((100, 176), "Nickel", fontsize=8)
    page.insert_text((300, 176), "remainder", fontsize=8)
    page.draw_line((90, 196), (520, 196))
    document.save(section_pdf)
    document.close()
    monkeypatch.chdir(tmp_path)

    manifest_path = extract_standard_assets_from_sections(
        tmp_path / "ASME-demo.pdf",
        [
            StandardPdfSection(
                index=1,
                title="Demo",
                standard_code="SA-1",
                start_page=10,
                end_page=10,
                output_path=str(section_pdf),
            )
        ],
    )

    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "TABLE 1 CHEMICAL REQUIREMENTS FOR NICKEL AND NICKEL ALLOY" in manifest_text


def test_standard_pdf_text_extraction_uses_masked_pdf_without_table_text(tmp_path: Path, monkeypatch) -> None:
    section_pdf = tmp_path / "section.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 50), "1. Scope", fontsize=10)
    page.insert_text((72, 68), "This paragraph should remain in the text output.", fontsize=10)
    _insert_centered_text(page, "TABLE 1", 110, fontsize=8)
    _insert_centered_text(page, "CHEMICAL REQUIREMENTS", 122, fontsize=8)
    page.draw_line((90, 138), (520, 138))
    page.insert_text((100, 156), "Element", fontsize=8)
    page.insert_text((300, 156), "Composition", fontsize=8)
    page.draw_line((90, 174), (520, 174))
    page.insert_text((100, 194), "Nickel", fontsize=8)
    page.insert_text((300, 194), "remainder", fontsize=8)
    page.draw_line((90, 214), (520, 214))
    document.save(section_pdf)
    document.close()
    monkeypatch.chdir(tmp_path)

    section = StandardPdfSection(
        index=1,
        title="Demo",
        standard_code="SA-1",
        start_page=10,
        end_page=10,
        output_path=str(section_pdf),
    )
    manifest_path = extract_standard_assets_from_sections(tmp_path / "ASME-demo.pdf", [section])
    assets = load_standard_assets_manifest(manifest_path)
    masked_sections = write_masked_text_pdfs(tmp_path / "ASME-demo.pdf", [section], assets)
    items = extract_standard_text_from_sections(masked_sections)
    text = "\n".join(item.get("text", "") for item in items)

    assert Path(masked_sections[0].text_pdf_path).exists()
    assert "This paragraph should remain in the text output." in text
    assert "TABLE 1" not in text
    assert "CHEMICAL REQUIREMENTS" not in text
    assert "Nickel" not in text
    assert "remainder" not in text


def test_standard_pdf_text_extraction_masks_captionless_small_tables(tmp_path: Path, monkeypatch) -> None:
    section_pdf = tmp_path / "section.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "1.2 Plates are available in three grades as follows:", fontsize=10)
    page.insert_text((90, 95), "Grade U.S. [SI]", fontsize=7)
    page.insert_text((210, 95), "Tensile Strength, ksi [MPa]", fontsize=7)
    page.insert_text((98, 108), "60 [415]", fontsize=7)
    page.insert_text((226, 108), "60-80 [415-550]", fontsize=7)
    page.insert_text((98, 121), "65 [450]", fontsize=7)
    page.insert_text((226, 121), "65-85 [450-585]", fontsize=7)
    page.insert_text((72, 150), "1.3 This paragraph should remain after masking.", fontsize=10)
    document.save(section_pdf)
    document.close()
    monkeypatch.chdir(tmp_path)

    section = StandardPdfSection(
        index=1,
        title="Demo",
        standard_code="SA-1",
        start_page=10,
        end_page=10,
        output_path=str(section_pdf),
    )
    masked_sections = write_masked_text_pdfs(tmp_path / "ASME-demo.pdf", [section], [])
    items = extract_standard_text_from_sections(masked_sections)
    text = "\n".join(item.get("text", "") for item in items)

    assert "Plates are available in three grades as follows:" in text
    assert "This paragraph should remain after masking." in text
    assert "Grade U.S. [SI]" not in text
    assert "Tensile Strength" not in text
    assert "60-80 [415-550]" not in text


def test_standard_pdf_mask_uses_rotated_page_coordinates(tmp_path: Path, monkeypatch) -> None:
    section_pdf = tmp_path / "section.pdf"
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((80, 120), "Rotated table text should be removed", fontsize=10)
    page.set_rotation(90)
    document.save(section_pdf)
    document.close()
    monkeypatch.chdir(tmp_path)

    section = StandardPdfSection(
        index=1,
        title="Demo",
        standard_code="SA-1",
        start_page=10,
        end_page=10,
        output_path=str(section_pdf),
    )
    asset = StandardAsset(
        index=1,
        asset_type="table",
        caption="TABLE 1 Rotated",
        section_index=1,
        standard_code="SA-1",
        section_page=1,
        source_page=10,
        bbox=(0.0, 0.0, 792.0, 612.0),
        image_path="table.png",
    )

    masked_sections = write_masked_text_pdfs(tmp_path / "ASME-demo.pdf", [section], [asset])

    with fitz.open(masked_sections[0].text_pdf_path) as masked:
        assert masked[0].get_text().strip() == ""


def test_processing_document_dir_mirrors_data_raw_relative_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("data") / "raw" / "产品标准" / "ASME-demo.pdf"

    assert processing_document_dir(source) == Path("data") / "processing" / "产品标准" / "ASME-demo"
