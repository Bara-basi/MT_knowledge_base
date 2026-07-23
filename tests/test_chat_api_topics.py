from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID

from app.api.v1 import chat
from app.schemas.chat import (
    ConversationTopicContextRequest,
    CreateConversationTopicRequest,
)


def test_create_topic_returns_fallback_summary_and_empty_messages(monkeypatch) -> None:
    async def fake_create_topic(**_kwargs):
        return {
            "topic_id": UUID("00000000-0000-0000-0000-000000000001"),
            "user_id": "user-1",
            "session_id": "session-1",
            "topic": "测试主题",
            "summary": "",
            "started_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "last_message_at": datetime.now(timezone.utc),
            "message_count": 0,
            "status": "active",
            "metadata": {},
        }

    monkeypatch.setattr(chat, "create_conversation_topic_record", fake_create_topic)

    response = asyncio.run(
        chat.create_topic(
            CreateConversationTopicRequest(
                user_id="user-1",
                session_id="session-1",
                topic="测试主题",
            )
        )
    )

    assert response.topic.summary == "无近期对话"
    assert response.messages == []
    assert "status" not in response.model_dump()


def test_get_topic_context_persists_received_summary(monkeypatch) -> None:
    topic_id = UUID("00000000-0000-0000-0000-000000000001")
    topic = {
        "topic_id": topic_id,
        "user_id": "user-1",
        "session_id": "session-1",
        "topic": "Test topic",
        "summary": "Previous summary",
        "started_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "last_message_at": datetime.now(timezone.utc),
        "message_count": 1,
        "status": "active",
        "metadata": {},
    }
    captured: dict = {}

    async def fake_get_context(**_kwargs):
        return {"topic": topic, "messages": []}

    async def fake_update_summary(**kwargs):
        captured.update(kwargs)
        return {"topic": {**topic, "summary": kwargs["summary"]}, "bound_message": {}}

    monkeypatch.setattr(chat, "get_conversation_topic_context", fake_get_context)
    monkeypatch.setattr(chat, "update_conversation_topic_summary", fake_update_summary)
    monkeypatch.setattr(chat, "remember_topic_selection", lambda **_kwargs: None)

    response = asyncio.run(
        chat.get_topic_context(
            ConversationTopicContextRequest(
                user_id="user-1",
                session_id="session-1",
                topic_id=topic_id,
                summary="Updated summary",
            )
        )
    )

    assert captured == {
        "topic_id": topic_id,
        "user_id": "user-1",
        "session_id": "session-1",
        "summary": "Updated summary",
    }
    assert response.topic.summary == "Updated summary"
