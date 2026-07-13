from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field
from pydantic import field_validator


def _looks_like_corrupted_text(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if "\ufffd" in text:
        return True
    non_space = "".join(ch for ch in text if not ch.isspace())
    return bool(non_space) and set(non_space) <= {"?"}


def _reject_corrupted_text(value: str | None) -> str | None:
    if value is not None and _looks_like_corrupted_text(value):
        raise ValueError(
            "text looks corrupted; please send request body as UTF-8 JSON instead of question marks"
        )
    return value


class ConversationTopic(BaseModel):
    topic_id: UUID
    user_id: str
    session_id: str
    topic: str
    summary: str
    started_at: datetime
    updated_at: datetime
    last_message_at: datetime
    message_count: int
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    user_id: str
    user_name: str | None = None
    session_id: str
    conversation_id: str
    topic_id: UUID | None = None
    create_time: datetime
    question: str
    answer: str


class CreateConversationTopicRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Feishu union_id.")
    session_id: str = Field(..., min_length=1, description="Feishu chat_id/session id.")
    topic: str = Field(..., min_length=1, description="Short topic name generated upstream.")
    summary: str = Field("", description="Optional initial topic summary.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("topic", "summary")
    @classmethod
    def reject_corrupted_text(cls, value: str) -> str:
        return _reject_corrupted_text(value) or ""


class CreateConversationTopicResponse(BaseModel):
    topic: ConversationTopic
    messages: list[ConversationMessage] = Field(default_factory=list)


class RecentConversationTopicsRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Feishu union_id.")
    session_id: str = Field(..., min_length=1, description="Feishu chat_id/session id.")
    k: int = Field(5, ge=1, le=50, description="Maximum recent active topics to return.")


class RecentConversationTopicsResponse(BaseModel):
    topics: list[ConversationTopic]
    count: int
    status: Literal["success"] = "success"


class ConversationTopicContextRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Feishu union_id.")
    session_id: str = Field(..., min_length=1, description="Feishu chat_id/session id.")
    topic_id: UUID
    k: int = Field(10, ge=1, le=50, description="Recent turns to return for the topic.")


class ConversationTopicContextResponse(BaseModel):
    topic: ConversationTopic
    messages: list[ConversationMessage]


class UpdateConversationTopicSummaryRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Feishu union_id.")
    session_id: str = Field(..., min_length=1, description="Feishu chat_id/session id.")
    topic_id: UUID
    summary: str = Field(..., description="New encrypted-at-rest topic summary.")
    topic: str | None = Field(None, description="Optional corrected short topic name.")
    conversation_id: str | None = Field(
        None,
        description="Optional chat conversation_id. When provided, the latest chat row is bound to topic_id.",
    )
    message_increment: int = Field(
        0,
        ge=0,
        le=100,
        description="How many turns this manual summary update represents. Defaults to 0 because chat answer persistence owns message counting.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("topic", "summary")
    @classmethod
    def reject_corrupted_text(cls, value: str | None) -> str | None:
        return _reject_corrupted_text(value)


class UpdateConversationTopicSummaryResponse(BaseModel):
    topic: ConversationTopic
    bound_message: ConversationMessage | None = None
    status: Literal["success"] = "success"
