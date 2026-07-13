from __future__ import annotations

import asyncio

from app.services import chat_records


class FakeFuture:
    def add_done_callback(self, _callback) -> None:
        return None


class FakeSummaryExecutor:
    def __init__(self) -> None:
        self.submitted: list[dict] = []

    def submit(self, func, **kwargs):
        self.submitted.append({"func": func, "kwargs": kwargs})
        return FakeFuture()


def test_chat_answer_records_topic_id(monkeypatch) -> None:
    answered_rows: list[dict] = []
    touched_topics: list[dict] = []
    fake_executor = FakeSummaryExecutor()

    monkeypatch.setattr(chat_records, "ensure_chat_messages_table", lambda: None)

    def fake_update_answer(**kwargs):
        answered_rows.append(kwargs)
        return kwargs

    monkeypatch.setattr(chat_records, "update_chat_answer", fake_update_answer)

    def fake_touch_topic(**kwargs):
        touched_topics.append(kwargs)
        return {**kwargs, "message_count": 10}

    monkeypatch.setattr(chat_records, "touch_conversation_topic_activity", fake_touch_topic)
    monkeypatch.setattr(chat_records, "_summary_executor", fake_executor)

    asyncio.run(
        chat_records.record_chat_answer(
            user_id="user-1",
            session_id="session-1",
            conversation_id="conversation-1",
            question="question",
            answer="answer",
            topic_id="00000000-0000-0000-0000-000000000001",
        )
    )

    assert answered_rows[0]["topic_id"] == "00000000-0000-0000-0000-000000000001"
    assert touched_topics == [
        {
            "topic_id": "00000000-0000-0000-0000-000000000001",
            "user_id": "user-1",
            "session_id": "session-1",
        }
    ]
    assert fake_executor.submitted == [
        {
            "func": chat_records._maybe_refresh_topic_summary,
            "kwargs": {
                "topic_id": "00000000-0000-0000-0000-000000000001",
                "user_id": "user-1",
                "session_id": "session-1",
                "message_count": 10,
            },
        }
    ]


def test_topic_summary_refresh_is_not_scheduled_before_interval(monkeypatch) -> None:
    fake_executor = FakeSummaryExecutor()
    monkeypatch.setattr(chat_records, "_summary_executor", fake_executor)

    chat_records._schedule_topic_summary_refresh(
        topic_id="00000000-0000-0000-0000-000000000001",
        user_id="user-1",
        session_id="session-1",
        message_count=9,
    )

    assert fake_executor.submitted == []
