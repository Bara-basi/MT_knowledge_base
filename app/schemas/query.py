from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    # Bound the public request so one call cannot exhaust the agent context.
    question: str = Field(..., min_length=1, max_length=8_000, description="User question.")
    user_id: str | None = Field(None, description="Caller or end-user id.")
    session_id: str | None = Field(None, description="Conversation/session id.")
    conversation_id: str | None = Field(None, description="External conversation id.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional business context passed through to the agent.",
    )
    service_id: str | None = None
    use_lark_document: bool = False
    format_type: Literal["markdown", "json"] = "markdown"
    additional_system_prompt: str = Field(
        "",
        max_length=20_000,
        description="Server-controlled extra task instructions for the final QA agent.",
    )
    task_input: str = Field(
        "",
        max_length=150_000,
        description="Structured input used only by an additional knowledge-base task.",
    )
    source: Literal["fastapi", "external"] = "fastapi"


class QueryResponse(BaseModel):
    question: str
    answer: str
    topic_id: str | None = None
    status: Literal["success"] = "success"
