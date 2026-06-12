from __future__ import annotations

from app.api.v1.retrieval import _sort_flow_results, _to_flow_chunk
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
        rerank_score=2.0,
        normalized_rerank_score=0.880797,
    )

    normal_payload = _to_flow_chunk(result).model_dump(exclude_none=True)
    debug_payload = _to_flow_chunk(result, debug=True).model_dump(exclude_none=True)

    assert "rerank_score" not in normal_payload
    assert "normalized_rerank_score" not in normal_payload
    assert debug_payload["rerank_score"] == 2.0
    assert debug_payload["normalized_rerank_score"] == 0.880797


def test_flow_results_sort_by_chunk_id_ascending() -> None:
    results = [
        RetrievalResult(id="doc_b_chunk_000010", score=0.5, content="", metadata={}),
        RetrievalResult(id="doc_a_chunk_000002", score=0.6, content="", metadata={}),
        RetrievalResult(id="doc_a_chunk_000001", score=0.7, content="", metadata={}),
    ]

    assert [result.id for result in _sort_flow_results(results)] == [
        "doc_a_chunk_000001",
        "doc_a_chunk_000002",
        "doc_b_chunk_000010",
    ]


def test_rerank_filter_normalizes_scores_and_cuts_low_score_tail() -> None:
    service = RetrievalService.__new__(RetrievalService)
    service.rerank_service = FakeRerankService(
        [
            RerankScore(id="a", score=8.0),
            RerankScore(id="b", score=-1.0),
            RerankScore(id="c", score=-3.0),
            RerankScore(id="d", score=-4.0),
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

    assert [result.id for result in ranked] == ["a"]
    assert ranked[0].rerank_score == 8.0
    assert round(ranked[0].normalized_rerank_score or 0, 4) == 0.9997


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
