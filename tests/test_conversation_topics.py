from __future__ import annotations

import asyncio

import pytest

from app.services.chat import topics
from app.services.privacy import decrypt_chat_text


def test_conversation_topic_create_encrypts_topic_and_summary(monkeypatch) -> None:
    captured_rows: list[dict] = []

    def fake_ensure_table() -> None:
        return None

    def fake_create_topic(**kwargs):
        captured_rows.append(kwargs)
        return {
            **kwargs,
            "started_at": None,
            "updated_at": None,
            "last_message_at": None,
            "message_count": 0,
            "status": "active",
        }

    monkeypatch.setattr(topics, "ensure_conversation_topics_table", fake_ensure_table)
    monkeypatch.setattr(topics, "create_conversation_topic", fake_create_topic)

    row = asyncio.run(
        topics.create_conversation_topic_record(
            user_id="user-1",
            session_id="session-1",
            topic="报价规则",
            summary="用户正在询问报价税率和利润计算。",
        )
    )

    stored = captured_rows[0]
    assert stored["topic"] != "报价规则"
    assert stored["summary"] != "用户正在询问报价税率和利润计算。"
    assert decrypt_chat_text(stored["topic"]) == "报价规则"
    assert decrypt_chat_text(stored["summary"]) == "用户正在询问报价税率和利润计算。"
    assert row["topic"] == "报价规则"
    assert row["summary"] == "用户正在询问报价税率和利润计算。"


def test_conversation_topic_update_encrypts_changed_text(monkeypatch) -> None:
    captured_rows: list[dict] = []

    def fake_ensure_table() -> None:
        return None

    def fake_update_topic(**kwargs):
        captured_rows.append(kwargs)
        return {
            "topic_id": kwargs["topic_id"],
            "user_id": "user-1",
            "session_id": "session-1",
            "topic": kwargs["topic"],
            "summary": kwargs["summary"],
            "started_at": None,
            "updated_at": None,
            "last_message_at": None,
            "message_count": 1,
            "status": "active",
            "metadata": {},
        }

    monkeypatch.setattr(topics, "ensure_conversation_topics_table", fake_ensure_table)
    monkeypatch.setattr(topics, "update_conversation_topic", fake_update_topic)

    row = asyncio.run(
        topics.update_conversation_topic_record(
            topic_id="00000000-0000-0000-0000-000000000001",
            topic="培训考试配置",
            summary="用户继续追问题库、考试发布和阅卷负责人。",
        )
    )

    stored = captured_rows[0]
    assert stored["topic"] != "培训考试配置"
    assert stored["summary"] != "用户继续追问题库、考试发布和阅卷负责人。"
    assert decrypt_chat_text(stored["topic"]) == "培训考试配置"
    assert decrypt_chat_text(stored["summary"]) == "用户继续追问题库、考试发布和阅卷负责人。"
    assert row["topic"] == "培训考试配置"
    assert row["summary"] == "用户继续追问题库、考试发布和阅卷负责人。"


def test_recent_conversation_topics_are_decrypted(monkeypatch) -> None:
    def fake_ensure_table() -> None:
        return None

    def fake_list_recent_topics(**_kwargs):
        return [
            {
                "topic_id": "topic-1",
                "user_id": "user-1",
                "session_id": "session-1",
                "topic": topics.encrypt_chat_text("报价规则"),
                "summary": topics.encrypt_chat_text("用户询问报价税率。"),
                "started_at": None,
                "updated_at": None,
                "last_message_at": None,
                "message_count": 2,
                "status": "active",
                "metadata": {},
            }
        ]

    monkeypatch.setattr(topics, "ensure_conversation_topics_table", fake_ensure_table)
    monkeypatch.setattr(topics, "list_recent_conversation_topics", fake_list_recent_topics)

    rows = asyncio.run(
        topics.list_recent_conversation_topic_records(
            user_id="user-1",
            session_id="session-1",
        )
    )

    assert rows[0]["topic"] == "报价规则"
    assert rows[0]["summary"] == "用户询问报价税率。"


def test_recent_conversation_topics_skip_corrupted_rows(monkeypatch) -> None:
    def fake_ensure_table() -> None:
        return None

    def fake_list_recent_topics(**_kwargs):
        return [
            {
                "topic_id": "topic-bad",
                "user_id": "user-1",
                "session_id": "session-1",
                "topic": topics.encrypt_chat_text("?????"),
                "summary": topics.encrypt_chat_text("?????"),
                "started_at": None,
                "updated_at": None,
                "last_message_at": None,
                "message_count": 1,
                "status": "active",
                "metadata": {},
            },
            {
                "topic_id": "topic-good",
                "user_id": "user-1",
                "session_id": "session-1",
                "topic": topics.encrypt_chat_text("报价规则"),
                "summary": topics.encrypt_chat_text("用户询问报价税率。"),
                "started_at": None,
                "updated_at": None,
                "last_message_at": None,
                "message_count": 2,
                "status": "active",
                "metadata": {},
            },
        ]

    monkeypatch.setattr(topics, "ensure_conversation_topics_table", fake_ensure_table)
    monkeypatch.setattr(topics, "list_recent_conversation_topics", fake_list_recent_topics)

    rows = asyncio.run(
        topics.list_recent_conversation_topic_records(
            user_id="user-1",
            session_id="session-1",
            limit=5,
        )
    )

    assert [row["topic_id"] for row in rows] == ["topic-good"]


def test_conversation_topic_rejects_corrupted_text(monkeypatch) -> None:
    monkeypatch.setattr(topics, "ensure_conversation_topics_table", lambda: None)

    with pytest.raises(ValueError):
        topics._create_conversation_topic_record_sync(
            user_id="user-1",
            session_id="session-1",
            topic="?????",
            summary="这里是总结",
            topic_id=None,
            started_at=None,
            metadata={},
        )
