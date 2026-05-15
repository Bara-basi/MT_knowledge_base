from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from app.db.milvus import MilvusCollectionConfig, ensure_chunk_collection, get_milvus_client
from app.services.embedding import EmbeddingService


@dataclass
class RetrievalResult:
    id: str
    score: float
    content: str
    metadata: dict[str, Any]
    document_id: str | None = None
    chunk_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "metadata": self.metadata,
        }


class RetrievalService:
    """Minimal vector recall service: query embedding -> Milvus search."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        client: MilvusClient | None = None,
        config: MilvusCollectionConfig | None = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.client = client or get_milvus_client()
        self.config = config or MilvusCollectionConfig()

    def search(
        self,
        query: str,
        limit: int = 5,
        bm25_model_file: str | Path | None = None,
    ) -> list[RetrievalResult]:
        query_vector = self.embedding_service.embed_query(query)
        bm25_model_path = bm25_model_file or default_bm25_model_file()
        query_sparse_vector = self.embedding_service.embed_bm25_query(query, bm25_model_path)
        ensure_chunk_collection(self.client, self.config)

        dense_request = AnnSearchRequest(
            data=[query_vector],
            anns_field="vector",
            param={
                "metric_type": self.config.metric_type,
                "params": {"ef": 64},
            },
            limit=limit,
        )
        sparse_request = AnnSearchRequest(
            data=[query_sparse_vector],
            anns_field="sparse_vector",
            param={
                "metric_type": "IP",
                "params": {"drop_ratio_search": 0.2},
            },
            limit=limit,
        )

        raw_results = self.client.hybrid_search(
            collection_name=self.config.name,
            reqs=[dense_request, sparse_request],
            ranker=RRFRanker(),
            limit=limit,
            output_fields=["id", "document_id", "chunk_index", "content", "metadata"],
        )
        return [_to_retrieval_result(hit) for hit in raw_results[0]]


def _to_retrieval_result(hit: dict[str, Any]) -> RetrievalResult:
    entity = hit.get("entity") or {}
    return RetrievalResult(
        id=str(hit.get("id") or entity.get("id")),
        score=float(hit.get("distance", hit.get("score", 0.0))),
        document_id=entity.get("document_id"),
        chunk_index=entity.get("chunk_index"),
        content=entity.get("content", ""),
        metadata=entity.get("metadata") or {},
    )


retrieval_service = RetrievalService()


def default_bm25_model_file() -> Path:
    return (
        Path("data")
        / "processing"
        / "订阅号运营SOP"
        / "embedding"
        / "订阅号运营SOP.bm25.json"
    )
