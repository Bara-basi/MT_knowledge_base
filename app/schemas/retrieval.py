from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FlowRetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question.")
    limit: int = Field(5, ge=1, le=50, description="Number of chunks to return.")
    document_name: str | None = Field(
        None,
        description="Processing document name used to resolve the default BM25 model.",
    )
    bm25_model_file: str | None = Field(
        None,
        description="Explicit BM25 model JSON path. Overrides document_name.",
    )
    recall_limit: int | None = Field(
        None,
        ge=1,
        le=200,
        description="Hybrid recall candidate count before reranking.",
    )
    rerank: bool = Field(True, description="Whether to use reranker after recall.")


class FlowRetrievedChunk(BaseModel):
    id: str
    order: int
    content: str
    structure: dict[str, Any] = Field(default_factory=dict)
    link: dict[str, str] | None = None
    img: list[str] | None = None


class FlowRetrievalResponse(BaseModel):
    query: str
    type: Literal["flow"] = "flow"
    count: int
    chunks: list[FlowRetrievedChunk]
