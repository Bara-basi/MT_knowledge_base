from __future__ import annotations

import json
from pathlib import Path

from app.db.milvus import MilvusCollectionConfig
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService
from scripts.ingestion import ingest_documents
from scripts.ingestion.ingest_documents import PreparedDocument, embed_prepared_document


def test_embed_chunks_returns_empty_for_empty_input() -> None:
    assert EmbeddingService().embed_chunks([]) == []


def test_embed_prepared_document_deletes_existing_vectors_for_empty_document(tmp_path) -> None:
    chunk_file = tmp_path / "empty.chunks.json"
    embedding_file = tmp_path / "empty.embeddings.json"
    chunk_file.write_text("[]", encoding="utf-8")
    prepared = PreparedDocument(
        file_path=tmp_path / "empty.pdf",
        txt_file=tmp_path / "empty.txt",
        chunk_file=chunk_file,
        embedding_file=embedding_file,
        chunks=[],
    )

    class EmbeddingServiceStub:
        def embed_chunk_file(self, chunk_file, output_file, **_kwargs):
            Path(output_file).write_text("[]", encoding="utf-8")
            return Path(output_file)

    class VectorStoreStub:
        def __init__(self):
            self.delete_file_ids = None

        def upsert_embedding_file(self, embedding_file, *, flush, delete_file_ids):
            self.delete_file_ids = list(delete_file_ids)
            return {"upsert_count": 0}

    vector_store = VectorStoreStub()
    result = embed_prepared_document(
        prepared,
        embedding_service=EmbeddingServiceStub(),
        vector_store_service=vector_store,
        flush=True,
        bm25_model=None,
        bm25_model_file=None,
    )

    assert result.upsert_count == 0
    assert json.loads(embedding_file.read_text(encoding="utf-8")) == []
    assert vector_store.delete_file_ids == [ingest_documents.document_file_id(prepared.file_path)]


def test_resolve_bm25_model_skips_empty_input_batch(tmp_path) -> None:
    prepared = PreparedDocument(
        file_path=tmp_path / "empty.pdf",
        txt_file=tmp_path / "empty.txt",
        chunk_file=tmp_path / "empty.chunks.json",
        embedding_file=tmp_path / "empty.embeddings.json",
        chunks=[],
    )

    model_file, model, action = ingest_documents.resolve_bm25_model(
        [prepared],
        embedding_service=EmbeddingService(),
        bm25_model_file=tmp_path / "missing.bm25.json",
        mode="auto",
    )

    assert model_file == tmp_path / "missing.bm25.json"
    assert model is None
    assert action == "skipped: no chunks in input batch"


def test_upsert_embedding_file_deletes_existing_file_id_before_upsert(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.services.vector_store.ensure_chunk_collection", lambda *_args, **_kwargs: {})

    embedding_file = tmp_path / "demo.embeddings.json"
    embedding_file.write_text(
        json.dumps(
            [
                {
                    "content": "new chunk",
                    "chunk_index": 0,
                    "metadata": {"file_id": "doc_demo"},
                    "embedding": [0.1, 0.2],
                    "bm25_embedding": {"1": 0.7},
                }
            ]
        ),
        encoding="utf-8",
    )

    class FakeClient:
        def __init__(self):
            self.calls = []

        def delete(self, **kwargs):
            self.calls.append(("delete", kwargs))
            return {"delete_count": 3}

        def upsert(self, **kwargs):
            self.calls.append(("upsert", kwargs))
            return {"upsert_count": len(kwargs["data"])}

        def flush(self, **kwargs):
            self.calls.append(("flush", kwargs))

    client = FakeClient()
    service = VectorStoreService(
        client=client,
        config=MilvusCollectionConfig(name="chunks", vector_dim=2),
    )

    result = service.upsert_embedding_file(embedding_file)

    assert result["delete_count"] == 3
    assert result["upsert_count"] == 1
    assert client.calls[0] == (
        "delete",
        {"collection_name": "chunks", "filter": 'file_id == "doc_demo"'},
    )
    assert client.calls[1][0] == "upsert"
    assert client.calls[2] == ("flush", {"collection_name": "chunks"})


def test_upsert_empty_embedding_file_can_delete_by_file_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.services.vector_store.ensure_chunk_collection", lambda *_args, **_kwargs: {})
    embedding_file = tmp_path / "empty.embeddings.json"
    embedding_file.write_text("[]", encoding="utf-8")

    class FakeClient:
        def __init__(self):
            self.calls = []

        def delete(self, **kwargs):
            self.calls.append(("delete", kwargs))
            return {"delete_count": 2}

        def flush(self, **kwargs):
            self.calls.append(("flush", kwargs))

    client = FakeClient()
    service = VectorStoreService(client=client, config=MilvusCollectionConfig(name="chunks"))

    result = service.upsert_embedding_file(embedding_file, delete_file_ids=["pdf_empty"])

    assert result["upsert_count"] == 0
    assert result["delete_count"] == 2
    assert client.calls == [
        ("delete", {"collection_name": "chunks", "filter": 'file_id == "pdf_empty"'}),
        ("flush", {"collection_name": "chunks"}),
    ]
