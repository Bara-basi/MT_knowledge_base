from __future__ import annotations

from app.services.chat import summary
from app.services.privacy import decrypt_chat_text, encrypt_chat_text


def test_topic_summary_refresh_decrypts_messages_and_encrypts_summary(monkeypatch) -> None:
    captured_update: dict = {}

    def fake_get_topic(**_kwargs):
        return {
            "topic_id": "topic-1",
            "user_id": "user-1",
            "session_id": "session-1",
            "topic": encrypt_chat_text("午餐评价"),
            "summary": encrypt_chat_text("用户之前询问午餐。"),
            "message_count": 10,
        }

    def fake_list_messages(**_kwargs):
        return [
            {
                "question": encrypt_chat_text("今天午餐怎么样？"),
                "answer": encrypt_chat_text("可以关注菜品反馈。"),
            }
        ]

    class FakeLLMClient:
        def chat(self, messages, **_kwargs):
            prompt = messages[-1]["content"]
            assert "用户之前询问午餐。" in prompt
            assert "今天午餐怎么样？" in prompt
            assert "可以关注菜品反馈。" in prompt
            return "用户正在讨论公司午餐质量和菜品反馈。"

    def fake_update_topic(**kwargs):
        captured_update.update(kwargs)
        return {
            "topic_id": kwargs["topic_id"],
            "user_id": "user-1",
            "session_id": "session-1",
            "topic": encrypt_chat_text("午餐评价"),
            "summary": kwargs["summary"],
            "message_count": 10,
        }

    monkeypatch.setattr(summary, "get_conversation_topic", fake_get_topic)
    monkeypatch.setattr(summary, "list_chat_messages_by_topic", fake_list_messages)
    monkeypatch.setattr(summary, "get_llm_client", lambda: FakeLLMClient())
    monkeypatch.setattr(summary, "update_conversation_topic", fake_update_topic)

    row = summary.refresh_topic_summary(
        topic_id="topic-1",
        user_id="user-1",
        session_id="session-1",
    )

    assert captured_update["message_increment"] == 0
    assert decrypt_chat_text(captured_update["summary"]) == "用户正在讨论公司午餐质量和菜品反馈。"
    assert row["summary"] == "用户正在讨论公司午餐质量和菜品反馈。"


def test_topic_summary_only_runs_every_ten_messages(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_refresh(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(summary, "refresh_topic_summary", fake_refresh)

    assert (
        summary.maybe_refresh_topic_summary(
            topic_id="topic-1",
            user_id="user-1",
            session_id="session-1",
            message_count=9,
        )
        is None
    )
    assert summary.maybe_refresh_topic_summary(
        topic_id="topic-1",
        user_id="user-1",
        session_id="session-1",
        message_count=10,
    ) == {"ok": True}
    assert calls == [
        {
            "topic_id": "topic-1",
            "user_id": "user-1",
            "session_id": "session-1",
        }
    ]
