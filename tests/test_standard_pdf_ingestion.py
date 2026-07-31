from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.db.minio import RawDocumentObject, build_minio_uri
from app.services import document_ingestion
from app.services.document_ingestion import prepare_documents
from app.services.parser import parser as document_parser
from app.services.parser import standard_pdf_splitter, unified_pdf_parser


def test_standard_pdf_detection_only_matches_unsplit_asme_parent() -> None:
    assert document_parser.is_standard_pdf_source(
        r"data\raw\产品标准\ASME-Sec-II-A-Vol1-2023.pdf"
    )
    assert not document_parser.is_standard_pdf_source(
        "minio://knowledge-raw-docs/%E4%BA%A7%E5%93%81%E6%A0%87%E5%87%86/other.pdf"
    )
    assert not document_parser.is_standard_pdf_source(
        "minio://knowledge-raw-docs/产品标准/国标/GB-T-12771.pdf"
    )
    assert not document_parser.is_standard_pdf_source(
        "minio://knowledge-raw-docs/%E4%BA%A7%E5%93%81%E6%A0%87%E5%87%86/"
        "ASME-demo%28%E5%88%87%E5%88%86%E7%89%88%29/SA-20.pdf"
    )
    assert not document_parser.is_standard_pdf_source("data/raw/process_guide/demo.pdf")


def test_document_discovery_prefers_generated_sections_over_asme_parent(
    monkeypatch,
) -> None:
    parent = RawDocumentObject(
        "knowledge-raw-docs",
        "产品标准/ASME-Sec-II-A-Vol1-2023.pdf",
    )
    section = RawDocumentObject(
        "knowledge-raw-docs",
        "产品标准/ASME-Sec-II-A-Vol1-2023(切分版)/SA-20.pdf",
    )
    monkeypatch.setattr(
        document_ingestion,
        "list_raw_document_objects",
        lambda *_args, **_kwargs: [parent, section],
    )

    assert document_ingestion.find_document_files("产品标准", recursive=True) == [section.uri]


def test_prepare_documents_requires_split_edition_when_parse_is_disabled(
    monkeypatch,
) -> None:
    source = "minio://knowledge-raw-docs/产品标准/ASME-demo.pdf"
    monkeypatch.setattr(
        document_ingestion,
        "resolve_pdf_document_sources",
        lambda *_args, **_kwargs: [source],
    )
    with pytest.raises(FileNotFoundError, match="split edition"):
        prepare_documents(
            source,
            image_analysis_workers=1,
            parse=False,
        )


def test_parser_resolves_asme_section_then_uses_only_unified_parser(
    tmp_path: Path,
    monkeypatch,
) -> None:
    local_pdf = tmp_path / "ASME-demo.pdf"
    local_pdf.write_bytes(b"parent")
    local_section_pdf = tmp_path / "SA-1.pdf"
    local_section_pdf.write_bytes(b"section")
    source = "minio://knowledge-raw-docs/产品标准/ASME-demo.pdf"
    section_source = "minio://knowledge-raw-docs/产品标准/ASME-demo(切分版)/SA-1.pdf"
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        document_parser,
        "_local_parser_path",
        lambda value: local_pdf if str(value) == source else local_section_pdf,
    )
    monkeypatch.setattr(document_parser, "synchronize_processed_assets", lambda *_args, **_kwargs: "")

    def fake_resolve(value, **kwargs):
        calls.append(("resolve", (value, kwargs)))
        return [section_source]

    def fake_parse(path, **kwargs):
        calls.append(("parse", (path, kwargs)))
        return [{"type": "paragraph", "style": "正文", "text": "parsed"}]

    monkeypatch.setattr(unified_pdf_parser, "resolve_pdf_document_sources", fake_resolve)
    monkeypatch.setattr(unified_pdf_parser, "parse_unified_pdf_document", fake_parse)

    items = document_parser.parse_document(source)

    assert items[0]["text"] == "parsed"
    assert [name for name, _payload in calls] == ["resolve", "parse"]
    assert calls[0][1][1]["local_path"] == local_pdf
    parsed_path, parse_kwargs = calls[1][1]
    assert parsed_path == local_section_pdf
    assert parse_kwargs["source_reference"] == section_source


def test_publish_existing_sections_persists_small_pdf_source_uris(
    tmp_path: Path,
    monkeypatch,
) -> None:
    section_pdf = tmp_path / "SA-1.pdf"
    section_pdf.write_bytes(b"section")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_pdf": "data/raw/产品标准/ASME-demo.pdf",
                "sections": [
                    {
                        "index": 1,
                        "title": "Demo",
                        "standard_code": "SA-1",
                        "start_page": 1,
                        "end_page": 2,
                        "output_path": str(section_pdf),
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    expected_uri = build_minio_uri(
        "knowledge-raw-docs",
        "产品标准/ASME-demo(切分版)/SA-1.pdf",
    )

    def fake_upload(source_file, sections, **kwargs):
        assert source_file == "data/raw/产品标准/ASME-demo.pdf"
        assert kwargs["source_reference"] == "minio://knowledge-raw-docs/产品标准/ASME-demo.pdf"
        return [replace(sections[0], source_uri=expected_uri)]

    monkeypatch.setattr(
        standard_pdf_splitter,
        "upload_standard_pdf_sections_to_minio",
        fake_upload,
    )

    published = standard_pdf_splitter.publish_standard_pdf_sections_from_manifest(
        manifest_path,
        source_reference="minio://knowledge-raw-docs/产品标准/ASME-demo.pdf",
    )

    assert published[0].source_uri == expected_uri
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["sections"][0]["source_uri"] == expected_uri


def test_prepare_documents_delegates_existing_split_pdf_to_generic_preparer(
    monkeypatch,
) -> None:
    parent_pdf = Path("data/raw/产品标准/ASME-demo.pdf")
    source_uri = build_minio_uri(
        "knowledge-raw-docs",
        "产品标准/ASME-demo(切分版)/SA-20.pdf",
    )
    calls: list[tuple[object, dict]] = []
    prepared = document_ingestion.PreparedDocument(
        file_path=source_uri,
        txt_file=Path("SA-20.txt"),
        chunk_file=Path("SA-20.chunks.json"),
        embedding_file=Path("SA-20.embeddings.json"),
        chunks=[],
    )
    monkeypatch.setattr(
        document_ingestion,
        "resolve_pdf_document_sources",
        lambda *_args, **_kwargs: [source_uri],
    )
    monkeypatch.setattr(
        document_ingestion,
        "prepare_document",
        lambda value, **kwargs: calls.append((value, kwargs)) or prepared,
    )

    results = prepare_documents(
        parent_pdf,
        image_analysis_workers=1,
        rebuild=False,
        parse=False,
    )

    assert results == [prepared]
    assert calls == [
        (
            source_uri,
            {
                "image_analysis_workers": 1,
                "rebuild": False,
                "parse": False,
            },
        )
    ]


def test_prepare_standard_sections_rejects_unpublished_local_evidence_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    parent_pdf = Path("data/raw/产品标准/ASME-demo.pdf")
    monkeypatch.setattr(
        document_ingestion,
        "resolve_pdf_document_sources",
        lambda *_args, **_kwargs: [parent_pdf],
    )

    with pytest.raises(FileNotFoundError, match="split edition"):
        prepare_documents(
            parent_pdf,
            image_analysis_workers=1,
            rebuild=False,
            parse=False,
        )
