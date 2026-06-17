from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

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
        delete_existing: bool = True,
        delete_file_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        records = load_embedding_records(embedding_file)
        rows = [self._build_milvus_row(record) for record in records]
        file_ids = _unique_file_ids(row["file_id"] for row in rows)
        if delete_file_ids is not None:
            file_ids = _unique_file_ids([*file_ids, *delete_file_ids])

        if not rows and not file_ids:
            return {
                "collection_name": self.config.name,
                "input_file": str(embedding_file),
                "upsert_count": 0,
                "delete_count": 0,
                "ids": [],
            }

        ensure_chunk_collection(self.client, self.config)
        delete_count = 0
        delete_results: list[dict[str, Any]] = []
        if delete_existing:
            for file_id in file_ids:
                delete_result = self.delete_by_file_id(file_id, flush=False)
                delete_results.append(delete_result)
                delete_count += int(delete_result.get("delete_count", 0) or 0)

        if not rows:
            if flush:
                self.client.flush(collection_name=self.config.name)
            return {
                "collection_name": self.config.name,
                "input_file": str(embedding_file),
                "upsert_count": 0,
                "delete_count": delete_count,
                "ids": [],
                "delete_results": delete_results,
            }

        result = self.client.upsert(collection_name=self.config.name, data=rows)
        if flush:
            self.client.flush(collection_name=self.config.name)
        return {
            "collection_name": self.config.name,
            "input_file": str(embedding_file),
            "upsert_count": _result_value(result, "upsert_count", len(rows)),
            "delete_count": delete_count,
            "ids": [row["id"] for row in rows],
            "delete_results": delete_results,
            "result": result,
        }

    def delete_by_file_id(self, file_id: str, *, flush: bool = True) -> dict[str, Any]:
        cleaned_file_id = str(file_id or "").strip()
        if not cleaned_file_id:
            return {
                "collection_name": self.config.name,
                "file_id": cleaned_file_id,
                "delete_count": 0,
            }

        ensure_chunk_collection(self.client, self.config)
        result = self.client.delete(
            collection_name=self.config.name,
            filter=f'file_id == "{_escape_milvus_string(cleaned_file_id)}"',
        )
        if flush:
            self.client.flush(collection_name=self.config.name)
        return {
            "collection_name": self.config.name,
            "file_id": cleaned_file_id,
            "delete_count": _result_value(result, "delete_count", 0),
            "result": result,
        }

    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        return self.client.query(
            collection_name=self.config.name,
            ids=ids,
            output_fields=["id", "file_id", "chunk_index", "content", "metadata"],
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
        file_id = record.get("file_id") or metadata.get("file_id")
        if not file_id:
            raise VectorStoreError("Embedding record metadata is missing file_id.")
        file_id = str(file_id)
        chunk_id = record.get("vector_id") or build_chunk_id(file_id, chunk_index)

        metadata.update(
            {
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "file_id": file_id,
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
            "file_id": file_id,
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


def build_file_id(metadata: dict[str, Any]) -> str:
    file_type = str(metadata.get("file_type") or "")
    file_name = str(metadata.get("file_name") or "")
    if not file_type or not file_name:
        raise VectorStoreError("Cannot build file_id without file_type and file_name.")
    file_stem = Path(file_name).stem
    digest = hashlib.sha1(file_stem.encode("utf-8")).hexdigest()[:12]
    return f"{file_type}_{digest}"


def build_chunk_id(file_id: str, chunk_index: int) -> str:
    return f"{file_id}_chunk_{chunk_index:06d}"


def normalize_sparse_vector(sparse_vector: dict[Any, Any]) -> dict[int, float]:
    normalized: dict[int, float] = {}
    for key, value in sparse_vector.items():
        weight = float(value)
        if weight <= 0:
            continue
        normalized[int(key)] = weight
    return normalized


def _unique_file_ids(file_ids: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for file_id in file_ids:
        cleaned = str(file_id or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def _escape_milvus_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _result_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


vector_store_service = VectorStoreService()
