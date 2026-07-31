from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    # The question is forwarded to several workflow/LLM nodes.  Bound it at
    # the public API boundary so one webhook request cannot exhaust the
    # workflow context window or tie up a worker for the full n8n timeout.
    question: str = Field(..., min_length=1, max_length=8_000, description="User question.")
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
    current_topic: str = "无近期对话"
    current_summary: str = "无近期对话"
    history_topics: list[dict[str, Any]] = Field(default_factory=list)
    source: Literal["fastapi"] = "fastapi"


class QueryResponse(BaseModel):
    question: str
    answer: str
    topic_id: str | None = None
    status: Literal["success"] = "success"
