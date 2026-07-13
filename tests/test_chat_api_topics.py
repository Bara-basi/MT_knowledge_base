from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID

from app.api.v1 import chat
from app.schemas.chat import CreateConversationTopicRequest


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
