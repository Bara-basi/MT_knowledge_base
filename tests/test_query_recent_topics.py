from __future__ import annotations

import asyncio

from app.api.v1 import query
from app.services.chat.topic_selection import consume_topic_selection, remember_topic_selection


def test_topic_context_for_n8n_sets_current_summary_and_history(monkeypatch) -> None:
    async def fake_list_recent_topics(**kwargs):
        assert kwargs["user_id"] == "user-1"
        assert kwargs["session_id"] == "session-1"
        return [
            {
                "topic_id": "00000000-0000-0000-0000-000000000001",
                "topic": "报价规则",
                "summary": "用户询问报价税率。",
            },
            {
                "topic_id": "00000000-0000-0000-0000-000000000002",
                "topic": "培训考试",
                "summary": "用户询问题库导入。",
            },
        ]

    monkeypatch.setattr(query, "list_recent_conversation_topic_records", fake_list_recent_topics)

    topic_context = asyncio.run(
        query.build_topic_context_for_n8n(
            user_id="user-1",
            session_id="session-1",
        )
    )

    assert topic_context == {
        "current_topic": "报价规则",
        "current_summary": "用户询问报价税率。",
        "history_topics": [
            {
                "topic_id": "00000000-0000-0000-0000-000000000001",
                "topic": "报价规则",
                "summary": "用户询问报价税率。",
            },
            {
                "topic_id": "00000000-0000-0000-0000-000000000002",
                "topic": "培训考试",
                "summary": "用户询问题库导入。",
            },
        ],
    }


def test_topic_context_for_n8n_falls_back_without_user_or_session() -> None:
    topic_context = asyncio.run(
        query.build_topic_context_for_n8n(
            user_id=None,
            session_id="session-1",
        )
    )

    assert topic_context == {
        "current_topic": "无近期对话",
        "current_summary": "无近期对话",
        "history_topics": [],
    }


def test_extract_topic_id_from_n8n_response() -> None:
    topic_id = "00000000-0000-0000-0000-000000000001"

    assert query._extract_topic_id({"answer": "ok", "topic_id": topic_id}) == topic_id
    assert query._extract_topic_id({"data": {"selected_topic_id": topic_id}}) == topic_id


def test_topic_selection_is_consumed_once() -> None:
    topic_id = "00000000-0000-0000-0000-000000000003"

    remember_topic_selection(
        user_id="user-1",
        session_id="session-1",
        topic_id=topic_id,
    )

    assert consume_topic_selection(user_id="user-1", session_id="session-1") == topic_id
    assert consume_topic_selection(user_id="user-1", session_id="session-1") is None
