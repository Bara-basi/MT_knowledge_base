from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app.schemas.retrieval import (
    FlowRetrievalRequest,
    FlowRetrievalResponse,
    FlowRetrievedChunk,
)
from app.services.retrieval import RetrievalResult, get_retrieval_service


router = APIRouter(prefix="/retrieval", tags=["retrieval"])

STRUCTURE_METADATA_KEYS = (
    "title",
    "chapter",
    "heading_3",
    "heading_4",
    "heading_5",
    "heading_6",
    "step",
)


@router.post(
    "/flow",
    response_model=FlowRetrievalResponse,
    response_model_exclude_none=True,
)
def retrieve_flow_chunks(request: FlowRetrievalRequest) -> FlowRetrievalResponse:
    """Recall and rerank chunks for process-style QA contexts."""
    bm25_model_file = _resolve_bm25_model_file(request)

    try:
        results = get_retrieval_service().search(
            request.query,
            limit=request.limit,
            bm25_model_file=bm25_model_file,
            recall_limit=request.recall_limit,
            rerank=request.rerank,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    chunks = [
        _to_flow_chunk(index, result)
        for index, result in enumerate(_sort_flow_results(results), start=1)
    ]
    return FlowRetrievalResponse(
        query=request.query,
        count=len(chunks),
        chunks=chunks,
    )


def _resolve_bm25_model_file(request: FlowRetrievalRequest) -> str | Path | None:
    if request.bm25_model_file:
        return request.bm25_model_file
    if request.document_name:
        return (
            Path("data")
            / "processing"
            / request.document_name
            / "embedding"
            / f"{request.document_name}.bm25.json"
        )
    return None


def _sort_flow_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    return sorted(
        results,
        key=lambda result: (
            str(result.metadata.get("title") or result.document_id or ""),
            result.chunk_index if result.chunk_index is not None else 10**9,
        ),
    )


def _to_flow_chunk(order: int, result: RetrievalResult) -> FlowRetrievedChunk:
    metadata = result.metadata or {}
    structure = _extract_structure(metadata)

    return FlowRetrievedChunk(
        order=order,
        content=result.content,
        structure=structure,
        link=_normalize_links(metadata.get("link")),
        img=_normalize_images(metadata.get("img")),
    )


def _extract_structure(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in STRUCTURE_METADATA_KEYS
        if key in metadata and _has_value(metadata[key])
    }


def _normalize_links(value: Any) -> dict[str, str] | None:
    if isinstance(value, dict):
        links = {
            str(key): str(url)
            for key, url in value.items()
            if _has_value(key) and _has_value(url)
        }
        return links or None

    if isinstance(value, list):
        links = {
            f"link_{index}": str(url)
            for index, url in enumerate(value, start=1)
            if _has_value(url)
        }
        return links or None

    return None


def _normalize_images(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    images = [str(item) for item in value if _has_value(item)]
    return images or None


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True
