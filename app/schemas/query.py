from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question.")
    user_id: str | None = Field(None, description="Caller or end-user id.")
    session_id: str | None = Field(None, description="Conversation/session id.")
    conversation_id: str | None = Field(None, description="External conversation id.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional business context passed through to n8n.",
    )


class N8nQueryRequest(BaseModel):
    question: str
    user_id: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: Literal["fastapi"] = "fastapi"


class QueryResponse(BaseModel):
    question: str
    answer: str
    status: Literal["success"] = "success"
