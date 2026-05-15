from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient

from app.db.milvus import (
    MilvusCollectionConfig,
    ensure_chunk_collection,
    get_milvus_client,
)


class VectorStoreError(RuntimeError):
    """Raised when embedding records cannot be written to the vector store."""


class VectorStoreService:
    """Milvus-backed vector storage for embedded document chunks."""

    def __init__(
        self,
        client: MilvusClient | None = None,
        config: MilvusCollectionConfig | None = None,
    ) -> None:
        self.client = client or get_milvus_client()
        self.config = config or MilvusCollectionConfig()

    def upsert_embedding_file(
        self,
        embedding_file: str | Path,
        *,
        flush: bool = True,
    ) -> dict[str, Any]:
        records = load_embedding_records(embedding_file)
        rows = [self._build_milvus_row(record) for record in records]
        if not rows:
            return {
                "collection_name": self.config.name,
                "input_file": str(embedding_file),
                "upsert_count": 0,
                "ids": [],
            }

        ensure_chunk_collection(self.client, self.config)
        result = self.client.upsert(collection_name=self.config.name, data=rows)
        if flush:
            self.client.flush(collection_name=self.config.name)
        return {
            "collection_name": self.config.name,
            "input_file": str(embedding_file),
            "upsert_count": result.get("upsert_count", len(rows)),
            "ids": [row["id"] for row in rows],
            "result": result,
        }

    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        return self.client.query(
            collection_name=self.config.name,
            ids=ids,
            output_fields=["id", "document_id", "chunk_index", "content", "metadata"],
        )

    def _build_milvus_row(self, record: dict[str, Any]) -> dict[str, Any]:
        vector = record.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise VectorStoreError("Embedding record is missing a non-empty embedding list.")

        sparse_vector = record.get("bm25_embedding")
        if not isinstance(sparse_vector, dict) or not sparse_vector:
            raise VectorStoreError(
                "Embedding record is missing a non-empty bm25_embedding object."
            )

        if len(vector) != self.config.vector_dim:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {self.config.vector_dim}, "
                f"got {len(vector)}."
            )

        chunk_index = record.get("chunk_index")
        if chunk_index is None:
            raise VectorStoreError("Embedding record is missing chunk_index.")

        try:
            chunk_index = int(chunk_index)
        except (TypeError, ValueError) as exc:
            raise VectorStoreError(f"Invalid chunk_index: {chunk_index!r}") from exc

        metadata = dict(record.get("metadata") or {})
        document_id = record.get("document_id") or build_document_id(metadata)
        chunk_id = record.get("vector_id") or build_chunk_id(document_id, chunk_index)

        metadata.update(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "embedding_model": record.get("embedding_model"),
                "embedding_dimension": len(vector),
                "bm25_model": record.get("bm25_model"),
                "bm25_language": record.get("bm25_language"),
                "bm25_dimension": record.get("bm25_dimension"),
            }
        )

        return {
            "id": chunk_id,
            "vector": vector,
            "sparse_vector": normalize_sparse_vector(sparse_vector),
            "content": record.get("content", ""),
            "document_id": document_id,
            "chunk_index": chunk_index,
            "metadata": metadata,
        }


def load_embedding_records(embedding_file: str | Path) -> list[dict[str, Any]]:
    path = Path(embedding_file)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise VectorStoreError(f"Embedding file must contain a JSON list: {path}")

    for item in data:
        if not isinstance(item, dict):
            raise VectorStoreError(f"Embedding item must be a JSON object: {path}")
    return data


def build_document_id(metadata: dict[str, Any]) -> str:
    source = str(metadata.get("source_file") or metadata.get("title") or "unknown")
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"


def build_chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}_chunk_{chunk_index:06d}"


def normalize_sparse_vector(sparse_vector: dict[Any, Any]) -> dict[int, float]:
    normalized: dict[int, float] = {}
    for key, value in sparse_vector.items():
        weight = float(value)
        if weight <= 0:
            continue
        normalized[int(key)] = weight
    return normalized


vector_store_service = VectorStoreService()
