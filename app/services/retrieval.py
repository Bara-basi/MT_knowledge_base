from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from app.core.config import settings
from app.db.milvus import (
    MilvusCollectionConfig,
    ensure_chunk_collection,
    get_milvus_client,
)
from app.services.embedding import EmbeddingService
from app.services.rerank import RerankCandidate, RerankService


@dataclass
class RetrievalResult:
    id: str
    score: float
    content: str
    metadata: dict[str, Any]
    document_id: str | None = None
    chunk_index: int | None = None
    recall_score: float | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "recall_score": self.recall_score,
            "rerank_score": self.rerank_score,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "metadata": self.metadata,
        }


class RetrievalService:
    """Hybrid recall service with optional cross-encoder reranking."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        rerank_service: RerankService | None = None,
        client: MilvusClient | None = None,
        config: MilvusCollectionConfig | None = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.rerank_service = rerank_service or RerankService()
        self.client = client or get_milvus_client()
        self.config = config or MilvusCollectionConfig()

    def warmup_models(self) -> None:
        self.embedding_service.warmup()
        self.rerank_service.warmup()

    def search(
        self,
        query: str,
        limit: int = 5,
        bm25_model_file: str | Path | None = None,
        recall_limit: int | None = None,
        rerank: bool = True,
    ) -> list[RetrievalResult]:
        recall_limit = recall_limit or max(
            limit * settings.retrieval_recall_multiplier,
            limit,
        )
        query_vector = self.embedding_service.embed_query(query)
        bm25_model_path = bm25_model_file or default_bm25_model_file()
        query_sparse_vector = self.embedding_service.embed_bm25_query(
            query,
            bm25_model_path,
        )
        ensure_chunk_collection(self.client, self.config)

        dense_request = AnnSearchRequest(
            data=[query_vector],
            anns_field="vector",
            param={
                "metric_type": self.config.metric_type,
                "params": {"ef": 64},
            },
            limit=recall_limit,
        )
        sparse_request = AnnSearchRequest(
            data=[query_sparse_vector],
            anns_field="sparse_vector",
            param={
                "metric_type": "IP",
                "params": {"drop_ratio_search": 0.2},
            },
            limit=recall_limit,
        )

        raw_results = self.client.hybrid_search(
            collection_name=self.config.name,
            reqs=[dense_request, sparse_request],
            ranker=RRFRanker(),
            limit=recall_limit,
            output_fields=["id", "document_id", "chunk_index", "content", "metadata"],
        )
        results = [_to_retrieval_result(hit) for hit in raw_results[0]]
        if not rerank or not results:
            return results[:limit]
        return self._rerank_results(query, results)[:limit]

    def _rerank_results(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        candidates = [
            RerankCandidate(id=result.id, content=result.content)
            for result in results
        ]
        rerank_scores = self.rerank_service.rerank(query, candidates)
        result_by_id = {result.id: result for result in results}
        ranked_results: list[RetrievalResult] = []

        for rerank_score in rerank_scores:
            result = result_by_id.get(rerank_score.id)
            if result is None:
                continue
            result.recall_score = result.score
            result.rerank_score = rerank_score.score
            result.score = rerank_score.score
            ranked_results.append(result)

        ranked_ids = {result.id for result in ranked_results}
        ranked_results.extend(
            result for result in results if result.id not in ranked_ids
        )
        return ranked_results


def _to_retrieval_result(hit: dict[str, Any]) -> RetrievalResult:
    entity = hit.get("entity") or {}
    return RetrievalResult(
        id=str(hit.get("id") or entity.get("id")),
        score=float(hit.get("distance", hit.get("score", 0.0))),
        recall_score=float(hit.get("distance", hit.get("score", 0.0))),
        document_id=entity.get("document_id"),
        chunk_index=entity.get("chunk_index"),
        content=entity.get("content", ""),
        metadata=entity.get("metadata") or {},
    )


retrieval_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    global retrieval_service
    if retrieval_service is None:
        retrieval_service = RetrievalService()
    return retrieval_service


def default_bm25_model_file() -> Path:
    return (
        Path("data")
        / "processing"
        / "订阅号运营SOP"
        / "embedding"
        / "订阅号运营SOP.bm25.json"
    )
