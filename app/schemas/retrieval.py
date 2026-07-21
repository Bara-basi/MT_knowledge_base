from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FlowRetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question.")
    limit: int = Field(
        15,
        ge=1,
        le=50,
        description="Maximum number of chunks to return.",
    )
    document_name: str | None = Field(
        None,
        description="Optional document hint kept for client compatibility; BM25 uses the global model by default.",
    )
    bm25_model_file: str | None = Field(
        None,
        description="Explicit BM25 model JSON path for debugging. Defaults to the global BM25 model.",
    )
    recall_limit: int | None = Field(
        None,
        ge=1,
        le=200,
        description="Hybrid recall candidate count before reranking.",
    )
    rerank: bool = Field(True, description="Whether to use reranker after recall.")
    debug: bool = Field(
        False,
        description="Whether to include retrieval debug scores in each chunk.",
    )


class FilteredFlowRetrievalRequest(FlowRetrievalRequest):
    limit: int = Field(
        8,
        ge=1,
        le=20,
        description="Maximum filtered chunks returned to the QA agent.",
    )
    file_path: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Exact document file_path supplied by graph retrieval.",
    )
    chunk_type: str | None = Field(
        None,
        max_length=64,
        description="Optional exact metadata.chunk_type filter.",
    )
    path_prefix: str | None = Field(
        None,
        max_length=500,
        description="Optional metadata.path prefix used to narrow the document section.",
    )


class FlowRetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    chunk_index: int | None = None
    chunk_type: str = ""
    file_name: str = ""
    file_path: str = ""
    path: str = ""
    links: list[dict[str, Any]] | None = None
    imgs: list[dict[str, Any]] | None = None
    rerank_score: float | None = None


class FlowRetrievalResponse(BaseModel):
    query: str
    type: Literal["flow"] = "flow"
    count: int
    chunks: list[FlowRetrievedChunk]
