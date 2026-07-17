from __future__ import annotations

import json
from pathlib import Path

from scripts.ingestion import ingest_documents
from scripts.ingestion.ingest_documents import PreparedDocument, embed_prepared_document, prepare_document


def test_prepare_document_reuses_existing_txt_when_rebuild_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source_file = tmp_path / "raw" / "demo.docx"
    source_file.parent.mkdir()
    source_file.write_text("placeholder", encoding="utf-8")

    txt_file = tmp_path / "data" / "processing" / "demo" / "txt" / "demo.txt"
    txt_file.parent.mkdir(parents=True)
    txt_file.write_text("[paragraph] [正文] existing parsed text", encoding="utf-8")

    def fail_parse_document(*_args, **_kwargs):
        raise AssertionError("parse_document should be skipped when txt exists")

    monkeypatch.setattr(ingest_documents, "parse_document", fail_parse_document)

    prepared = prepare_document(source_file, image_analysis_workers=1, rebuild=False)

    assert prepared.txt_file == Path("data") / "processing" / "demo" / "txt" / "demo.txt"
    assert prepared.chunk_file.exists()
    assert len(prepared.chunks) == 1


def test_prepare_document_rebuilds_chunks_from_existing_txt_when_parse_disabled(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source_file = tmp_path / "raw" / "demo.docx"
    source_file.parent.mkdir()
    source_file.write_text("placeholder", encoding="utf-8")

    txt_file = tmp_path / "data" / "processing" / "demo" / "txt" / "demo.txt"
    txt_file.parent.mkdir(parents=True)
    txt_file.write_text("[paragraph] [正文] existing parsed text", encoding="utf-8")

    chunk_file = tmp_path / "data" / "processing" / "demo" / "chunk" / "demo.chunks.json"
    chunk_file.parent.mkdir(parents=True)
    chunk_file.write_text(
        json.dumps([{"content": "stale chunk", "metadata": {"file_id": "doc_demo"}, "chunk_index": 0}]),
        encoding="utf-8",
    )

    def fail_parse_document(*_args, **_kwargs):
        raise AssertionError("parse_document should be skipped when parse is disabled")

    monkeypatch.setattr(ingest_documents, "parse_document", fail_parse_document)

    prepared = prepare_document(
        source_file,
        image_analysis_workers=1,
        rebuild=True,
        parse=False,
    )

    assert len(prepared.chunks) == 1
    assert prepared.chunks[0].content == "existing parsed text"
    assert "stale chunk" not in chunk_file.read_text(encoding="utf-8")


def test_prepare_document_reuses_existing_chunks_when_rebuild_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source_file = tmp_path / "raw" / "demo.docx"
    source_file.parent.mkdir()
    source_file.write_text("placeholder", encoding="utf-8")

    chunk_file = tmp_path / "data" / "processing" / "demo" / "chunk" / "demo.chunks.json"
    chunk_file.parent.mkdir(parents=True)
    chunk_file.write_text(
        json.dumps([{"content": "existing chunk", "metadata": {"file_id": "doc_demo"}, "chunk_index": 0}]),
        encoding="utf-8",
    )

    def fail_parse_document(*_args, **_kwargs):
        raise AssertionError("parse_document should be skipped when chunk file exists")

    monkeypatch.setattr(ingest_documents, "parse_document", fail_parse_document)

    prepared = prepare_document(source_file, image_analysis_workers=1, rebuild=False)

    assert len(prepared.chunks) == 1
    assert prepared.chunks[0].content == "existing chunk"


def test_embed_prepared_document_reuses_existing_embedding_when_rebuild_disabled(tmp_path) -> None:
    chunk_file = tmp_path / "demo.chunks.json"
    embedding_file = tmp_path / "demo.embeddings.json"
    chunk_file.write_text("[]", encoding="utf-8")
    embedding_file.write_text("[]", encoding="utf-8")
    prepared = PreparedDocument(
        file_path=tmp_path / "demo.docx",
        txt_file=tmp_path / "demo.txt",
        chunk_file=chunk_file,
        embedding_file=embedding_file,
        chunks=[],
    )

    class EmbeddingServiceStub:
        def embed_chunk_file(self, *_args, **_kwargs):
            raise AssertionError("embedding should be skipped when embedding file exists")

    result = embed_prepared_document(
        prepared,
        embedding_service=EmbeddingServiceStub(),
        vector_store_service=None,
        flush=True,
        bm25_model=None,
        bm25_model_file=None,
        rebuild=False,
    )

    assert result.embedding_file == embedding_file
    assert result.upsert_count == 0


def test_embed_prepared_document_force_upserts_migrated_metadata(tmp_path) -> None:
    chunk_file = tmp_path / "demo.chunks.json"
    embedding_file = tmp_path / "demo.embeddings.json"
    chunk_file.write_text("[]", encoding="utf-8")
    embedding_file.write_text("[]", encoding="utf-8")
    prepared = PreparedDocument(
        file_path="minio://knowledge-raw-docs/产品标准/demo(切分版)/SA-1.pdf",
        txt_file=tmp_path / "demo.txt",
        chunk_file=chunk_file,
        embedding_file=embedding_file,
        chunks=[],
        force_upsert=True,
    )

    class EmbeddingServiceStub:
        def embed_chunk_file(self, *_args, **_kwargs):
            raise AssertionError("existing embedding should be reused")

    class VectorStoreStub:
        def upsert_embedding_file(self, *_args, **_kwargs):
            return {"upsert_count": 1}

    result = embed_prepared_document(
        prepared,
        embedding_service=EmbeddingServiceStub(),
        vector_store_service=VectorStoreStub(),
        flush=True,
        bm25_model=None,
        bm25_model_file=None,
        rebuild=False,
        skip_existing_upsert=True,
    )

    assert result.upsert_count == 1
    assert not result.upsert_skipped
