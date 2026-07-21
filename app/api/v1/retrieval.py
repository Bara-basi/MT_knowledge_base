from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from fastapi import APIRouter, HTTPException

from app.schemas.retrieval import (
    FilteredFlowRetrievalRequest,
    FlowRetrievalRequest,
    FlowRetrievalResponse,
    FlowRetrievedChunk,
)
from app.services.retrieval import RetrievalResult, get_retrieval_service


router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post(
    "/flow",
    response_model=FlowRetrievalResponse,
    response_model_exclude_none=True,
)
def retrieve_flow_chunks(request: FlowRetrievalRequest) -> FlowRetrievalResponse:
    """Recall and rerank chunks for process-style QA contexts."""
    return _retrieve_flow_chunks(request)


@router.post(
    "/filtered",
    response_model=FlowRetrievalResponse,
    response_model_exclude_none=True,
)
def retrieve_filtered_flow_chunks(
    request: FilteredFlowRetrievalRequest,
) -> FlowRetrievalResponse:
    """Hybrid retrieval constrained to a graph-supplied document path."""

    filter_expression = build_agent_metadata_filter(request)
    print(
        "[retrieval] filtered request "
        f"query={request.query!r} file_path={request.file_path!r} "
        f"chunk_type={request.chunk_type!r} path_prefix={request.path_prefix!r}",
        flush=True,
    )
    bm25_model_file = _resolve_bm25_model_file(request)
    try:
        results = get_retrieval_service().search(
            request.query,
            limit=request.limit,
            bm25_model_file=bm25_model_file,
            recall_limit=request.recall_limit,
            rerank=request.rerank,
            filter_expression=filter_expression,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Filtered retrieval failed: {exc}") from exc

    chunks = [_to_flow_chunk(result, debug=request.debug) for result in results]
    return FlowRetrievalResponse(
        query=request.query,
        count=len(chunks),
        chunks=chunks,
    )


def build_agent_metadata_filter(request: FilteredFlowRetrievalRequest) -> str:
    clauses = [f'metadata["file_path"] == {_milvus_string(request.file_path)}']
    if request.chunk_type:
        clauses.append(f'metadata["chunk_type"] == {_milvus_string(request.chunk_type)}')
    if request.path_prefix:
        prefix = request.path_prefix.replace("%", "\\%").replace("_", "\\_")
        clauses.append(f'metadata["path"] like {_milvus_string(prefix + "%")}')
    return " and ".join(clauses)


def _milvus_string(value: str) -> str:
    return json.dumps(str(value).strip(), ensure_ascii=False)


def _retrieve_flow_chunks(request: FlowRetrievalRequest) -> FlowRetrievalResponse:
    print(
        "[retrieval] flow request "
        f"query={request.query!r} limit={request.limit} "
        f"document_name={request.document_name!r} "
        f"bm25_model_file={request.bm25_model_file!r} "
        f"recall_limit={request.recall_limit} rerank={request.rerank} "
        f"debug={request.debug}",
        flush=True,
    )
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
        print(f"[retrieval] flow failed status=404 error={exc}", flush=True)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        print(f"[retrieval] flow failed status=400 error={exc}", flush=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"[retrieval] flow failed status=500 error={exc}", flush=True)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    chunks = [
        _to_flow_chunk(result, debug=request.debug)
        for result in _sort_flow_results(results)
    ]
    print(
        f"[retrieval] flow success query={request.query!r} count={len(chunks)}",
        flush=True,
    )
    return FlowRetrievalResponse(
        query=request.query,
        count=len(chunks),
        chunks=chunks,
    )


def _resolve_bm25_model_file(request: FlowRetrievalRequest) -> str | Path | None:
    if request.bm25_model_file:
        return request.bm25_model_file
    return None


def _sort_flow_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    return sorted(results, key=lambda result: result.id)


def _to_flow_chunk(
    result: RetrievalResult,
    debug: bool = False,
) -> FlowRetrievedChunk:
    metadata = result.metadata or {}

    return FlowRetrievedChunk(
        chunk_id=result.id,
        content=result.content,
        chunk_index=_chunk_index(result, metadata),
        chunk_type=str(metadata.get("chunk_type") or ""),
        file_name=str(metadata.get("file_name") or ""),
        file_path=str(metadata.get("file_path") or ""),
        path=str(metadata.get("path") or ""),
        links=_normalize_links(metadata.get("links")),
        imgs=_normalize_images(metadata.get("imgs")),
        rerank_score=result.rerank_score if debug else None,
    )


def _chunk_index(result: RetrievalResult, metadata: dict[str, Any]) -> int | None:
    value = (
        result.chunk_index
        if result.chunk_index is not None
        else metadata.get("chunk_index")
    )
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_links(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    links = [
        _normalize_link_item(item)
        for item in value
    ]
    links = [item for item in links if item is not None]
    return links or None


def _normalize_images(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    images = [
        _normalize_image_item(item)
        for item in value
    ]
    images = [item for item in images if item is not None]
    return images or None


def _normalize_link_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    link_path = item.get("link_path")
    link_name = item.get("link_name")
    if not _has_value(link_path):
        return None
    return {
        "index": _asset_index(item),
        "link_name": str(link_name or link_path),
        "link_path": str(link_path),
    }


def _normalize_image_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    img_path = item.get("img_path")
    img_name = item.get("img_name")
    if not _has_value(img_path):
        return None
    return {
        "index": _asset_index(item),
        "img_name": str(img_name or Path(str(img_path)).name),
        "img_path": str(img_path),
    }


def _asset_index(item: dict[str, Any]) -> int:
    try:
        return int(item.get("index", 0))
    except (TypeError, ValueError):
        return 0


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True
