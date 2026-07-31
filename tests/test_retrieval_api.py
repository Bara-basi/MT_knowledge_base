from __future__ import annotations

from app.api.v1.retrieval import _to_flow_chunk, build_agent_metadata_filter
from app.schemas.retrieval import FilteredFlowRetrievalRequest
from app.services.rerank import RerankScore
from app.services.retrieval import RetrievalResult, RetrievalService


class FakeRerankService:
    def __init__(self, scores: list[RerankScore]) -> None:
        self.scores = scores

    def rerank(self, query, candidates):
        return self.scores


def test_flow_chunk_returns_only_n8n_context_fields() -> None:
    result = RetrievalResult(
        id="doc_demo_chunk_000002",
        score=0.8,
        content='<img index="7">图片：示例</img>\n<a index="8"></a>',
        file_id="doc_demo",
        chunk_index=2,
        metadata={
            "chunk_index": 2,
            "chunk_type": "text",
            "file_name": "demo.docx",
            "file_path": "data/raw/demo.docx",
            "path": "章节\\小节",
            "imgs": [
                {
                    "index": 7,
                    "img_name": "image_0007.png",
                    "img_path": "data/processing/demo/img/image_0007.png",
                }
            ],
            "links": [
                {
                    "index": 8,
                    "link_name": "官网链接",
                    "link_path": "https://example.com",
                }
            ],
            "file_id": "doc_demo",
            "extra": "not returned",
        },
    )

    payload = _to_flow_chunk(result).model_dump(exclude_none=True)

    assert payload == {
        "chunk_id": "doc_demo_chunk_000002",
        "content": '<img index="7">图片：示例</img>\n<a index="8"></a>',
        "chunk_index": 2,
        "chunk_type": "text",
        "file_name": "demo.docx",
        "file_path": "data/raw/demo.docx",
        "path": "章节\\小节",
        "links": [
            {
                "index": 8,
                "link_name": "官网链接",
                "link_path": "https://example.com",
            }
        ],
        "imgs": [
            {
                "index": 7,
                "img_name": "image_0007.png",
                "img_path": "data/processing/demo/img/image_0007.png",
            }
        ],
    }


def test_flow_chunk_includes_scores_only_when_debug_enabled() -> None:
    result = RetrievalResult(
        id="doc_demo_chunk_000002",
        score=0.88,
        content="debug",
        metadata={},
        rerank_score=0.88,
    )

    normal_payload = _to_flow_chunk(result).model_dump(exclude_none=True)
    debug_payload = _to_flow_chunk(result, debug=True).model_dump(exclude_none=True)

    assert "rerank_score" not in normal_payload
    assert debug_payload["rerank_score"] == 0.88


def test_rerank_filter_uses_raw_scores_and_cuts_low_score_tail() -> None:
    service = RetrievalService.__new__(RetrievalService)
    service.rerank_service = FakeRerankService(
        [
            RerankScore(id="a", score=0.95),
            RerankScore(id="b", score=0.91),
            RerankScore(id="c", score=0.2),
            RerankScore(id="d", score=0.1),
        ]
    )
    results = [
        RetrievalResult(id="a", score=0.9, content="a", metadata={}),
        RetrievalResult(id="b", score=0.8, content="b", metadata={}),
        RetrievalResult(id="c", score=0.7, content="c", metadata={}),
        RetrievalResult(id="d", score=0.6, content="d", metadata={}),
    ]

    ranked = service._filter_reranked_results(
        service._rerank_results("query", results),
        max_results=15,
    )

    assert [result.id for result in ranked] == ["a", "b"]
    assert [result.rerank_score for result in ranked] == [0.95, 0.91]


def test_rerank_limit_is_maximum_after_filters() -> None:
    service = RetrievalService.__new__(RetrievalService)
    service.rerank_service = FakeRerankService(
        [
            RerankScore(id="a", score=1.0),
            RerankScore(id="b", score=0.99),
            RerankScore(id="c", score=0.98),
            RerankScore(id="d", score=0.97),
            RerankScore(id="e", score=0.96),
            RerankScore(id="f", score=0.95),
        ]
    )
    results = [
        RetrievalResult(id="a", score=0.9, content="a", metadata={}),
        RetrievalResult(id="b", score=0.8, content="b", metadata={}),
        RetrievalResult(id="c", score=0.7, content="c", metadata={}),
        RetrievalResult(id="d", score=0.6, content="d", metadata={}),
        RetrievalResult(id="e", score=0.5, content="e", metadata={}),
        RetrievalResult(id="f", score=0.4, content="f", metadata={}),
    ]

    ranked = service._filter_reranked_results(
        service._rerank_results("query", results),
        max_results=2,
    )

    assert [result.id for result in ranked] == ["a", "b"]


def test_agent_metadata_filter_uses_allowlisted_structured_fields() -> None:
    request = FilteredFlowRetrievalRequest(
        query="A213 化学成分",
        file_path='minio://knowledge-raw-docs/A213 "special".pdf',
        chunk_type="table",
        path_prefix="Chemical_Requirements%",
    )

    expression = build_agent_metadata_filter(request)

    assert 'metadata["file_path"] == "minio://knowledge-raw-docs/A213 \\"special\\".pdf"' in expression
    assert 'metadata["chunk_type"] == "table"' in expression
    assert 'metadata["path"] like "Chemical\\\\_Requirements\\\\%%"' in expression


def test_retrieval_service_applies_same_filter_to_dense_and_sparse_requests() -> None:
    class FakeEmbeddingService:
        def embed_query(self, query):
            return [0.1, 0.2]

        def embed_bm25_query(self, query, model_path):
            return {1: 0.5}

    class FakeClient:
        def hybrid_search(self, **kwargs):
            self.kwargs = kwargs
            return [[]]

    service = RetrievalService.__new__(RetrievalService)
    service.embedding_service = FakeEmbeddingService()
    service.rerank_service = FakeRerankService([])
    service.client = FakeClient()
    service.config = type(
        "Config",
        (),
        {"name": "test", "metric_type": "COSINE"},
    )()
    service._dense_search_ef = lambda recall_limit: 64

    from unittest.mock import patch

    with patch("app.services.retrieval.ensure_chunk_collection"):
        service.search(
            "query",
            limit=2,
            recall_limit=3,
            rerank=False,
            filter_expression='metadata["file_path"] == "demo.pdf"',
        )

    requests = service.client.kwargs["reqs"]
    assert len(requests) == 2
    assert all(request.filter == 'metadata["file_path"] == "demo.pdf"' for request in requests)
