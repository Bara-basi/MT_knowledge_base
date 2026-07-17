from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.db.minio import RawDocumentObject, build_minio_uri
from app.services.chunking.splitter import save_chunks, split_items
from app.services import document_ingestion
from app.services.document_ingestion import prepare_document, prepare_documents
from app.services.parser import parser as document_parser
from app.services.parser import standard_pdf_parser, standard_pdf_splitter
from app.services.parser.paths import processing_document_dir


def test_standard_pdf_detection_covers_product_standard_paths_and_excludes_sections() -> None:
    assert document_parser.is_standard_pdf_source(
        r"data\raw\产品标准\ASME-Sec-II-A-Vol1-2023.pdf"
    )
    assert document_parser.is_standard_pdf_source(
        "minio://knowledge-raw-docs/%E4%BA%A7%E5%93%81%E6%A0%87%E5%87%86/other.pdf"
    )
    assert not document_parser.is_standard_pdf_source(
        "minio://knowledge-raw-docs/%E4%BA%A7%E5%93%81%E6%A0%87%E5%87%86/"
        "ASME-demo%28%E5%88%87%E5%88%86%E7%89%88%29/SA-20.pdf"
    )
    assert not document_parser.is_standard_pdf_source("data/raw/process_guide/demo.pdf")


def test_document_discovery_keeps_parent_standard_and_excludes_generated_sections(
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

    assert document_ingestion.find_document_files("产品标准", recursive=True) == [parent.uri]


def test_single_document_preparation_rejects_parent_standard_pdf() -> None:
    with pytest.raises(ValueError, match="prepare_documents"):
        prepare_document(
            "minio://knowledge-raw-docs/产品标准/ASME-demo.pdf",
            image_analysis_workers=1,
        )


def test_parser_splits_before_passing_sections_to_standard_parser(tmp_path: Path, monkeypatch) -> None:
    local_pdf = tmp_path / "ASME-demo.pdf"
    local_pdf.write_bytes(b"pdf")
    source = "minio://knowledge-raw-docs/产品标准/ASME-demo.pdf"
    section = standard_pdf_splitter.StandardPdfSection(
        index=1,
        title="Demo",
        standard_code="SA-1",
        start_page=1,
        end_page=2,
        output_path=str(tmp_path / "SA-1.pdf"),
        source_uri="minio://knowledge-raw-docs/产品标准/ASME-demo(切分版)/SA-1.pdf",
    )
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(document_parser, "_local_parser_path", lambda _source: local_pdf)

    def fake_split(path, **kwargs):
        calls.append(("split", (path, kwargs)))
        return [section]

    def fake_parse(sections):
        calls.append(("parse", sections))
        return [{"type": "paragraph", "style": "正文", "text": "parsed"}]

    monkeypatch.setattr(standard_pdf_splitter, "split_standard_pdf_document", fake_split)
    monkeypatch.setattr(standard_pdf_parser, "parse_standard_pdf_sections", fake_parse)

    items = document_parser.parse_document(source)

    assert items[0]["text"] == "parsed"
    assert [name for name, _payload in calls] == ["split", "parse"]
    split_path, split_kwargs = calls[0][1]
    assert split_path == local_pdf
    assert split_kwargs["source_reference"] == source
    assert calls[1][1] == [section]


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


def test_prepare_standard_sections_rewrites_stale_chunk_source_to_published_small_pdf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    parent_pdf = Path("data/raw/产品标准/ASME-demo.pdf")
    section_name = "SA-20 - Demo"
    processing_dir = processing_document_dir(parent_pdf)
    section_dir = processing_dir / section_name
    txt_file = section_dir / "txt" / f"{section_name}.txt"
    section_pdf = section_dir / "pdf" / f"{section_name}.pdf"
    chunk_file = section_dir / "chunk" / f"{section_name}.chunks.json"
    embedding_file = section_dir / "embedding" / f"{section_name}.embeddings.json"
    txt_file.parent.mkdir(parents=True)
    section_pdf.parent.mkdir(parents=True)
    section_pdf.write_bytes(b"section")
    txt_file.write_text(
        "[paragraph] [标题] SA-20 Demo\n"
        "[paragraph] [正文] This is enough standard text to create a source chunk.\n",
        encoding="utf-8",
    )
    source_uri = build_minio_uri(
        "knowledge-raw-docs",
        "产品标准/ASME-demo(切分版)/SA-20 - Demo.pdf",
    )
    (processing_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sections": [
                    {
                        "output_path": str(section_pdf),
                        "source_uri": source_uri,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    stale_chunks = split_items(
        [{"type": "paragraph", "style": "正文", "text": "stale parent source text"}],
        source_file=parent_pdf,
    )
    save_chunks(stale_chunks, chunk_file)
    embedding_file.parent.mkdir(parents=True)
    embedding_file.write_text(
        json.dumps([{"metadata": {"file_path": str(parent_pdf)}}]),
        encoding="utf-8",
    )

    prepared = prepare_documents(
        parent_pdf,
        image_analysis_workers=1,
        rebuild=False,
        parse=False,
    )

    assert len(prepared) == 1
    assert prepared[0].file_path == source_uri
    assert prepared[0].chunks
    assert prepared[0].force_upsert
    assert not embedding_file.exists()
    assert all(chunk.metadata["file_path"] == source_uri for chunk in prepared[0].chunks)
    assert all(chunk.metadata["file_name"] == f"{section_name}.pdf" for chunk in prepared[0].chunks)


def test_prepare_standard_sections_rejects_unpublished_local_evidence_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    parent_pdf = Path("data/raw/产品标准/ASME-demo.pdf")
    processing_dir = processing_document_dir(parent_pdf)
    txt_file = processing_dir / "SA-1" / "txt" / "SA-1.txt"
    txt_file.parent.mkdir(parents=True)
    txt_file.write_text("[paragraph] [正文] demo standard text", encoding="utf-8")
    (processing_dir / "manifest.json").write_text(
        json.dumps(
            {"sections": [{"output_path": str(processing_dir / "SA-1" / "pdf" / "SA-1.pdf")}]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        document_ingestion,
        "publish_standard_pdf_sections_from_manifest",
        lambda *_args, **_kwargs: [],
    )

    with pytest.raises(ValueError, match="source_uri after publication"):
        prepare_documents(
            parent_pdf,
            image_analysis_workers=1,
            rebuild=False,
            parse=False,
        )
