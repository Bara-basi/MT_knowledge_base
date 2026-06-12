from __future__ import annotations

import asyncio
import json

from app.api.v1 import feishu
from app.schemas.query import QueryResponse


def test_feishu_feedback_card_updates_until_final_answer(monkeypatch) -> None:
    sent_messages: list[tuple[str | None, str | None, str]] = []
    updates: list[tuple[str, str]] = []

    async def fake_send_message(
        *,
        chat_id: str | None,
        reply_to_message_id: str | None,
        markdown_text: str,
        **_kwargs,
    ) -> str:
        sent_messages.append((chat_id, reply_to_message_id, markdown_text))
        return "feedback-message-id"

    async def fake_update_message(message_id: str, markdown_text: str, **_kwargs) -> bool:
        updates.append((message_id, markdown_text))
        return True

    async def fake_ask_knowledge_base(_request):
        await asyncio.sleep(0.16)
        return QueryResponse(question="question", answer="final answer")

    async def fake_create_chat_record(**_kwargs) -> None:
        return None

    async def fake_record_chat_answer(**_kwargs) -> None:
        return None

    monkeypatch.setattr(feishu, "_feedback_interval_seconds", 0.05)
    monkeypatch.setattr(feishu, "_load_feedback_texts", lambda: ["step 1", "step 2", "step 3"])
    monkeypatch.setattr(feishu, "_try_send_feishu_markdown_message", fake_send_message)
    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)
    monkeypatch.setattr(feishu, "ask_knowledge_base", fake_ask_knowledge_base)
    monkeypatch.setattr(feishu, "create_chat_record", fake_create_chat_record)
    monkeypatch.setattr(feishu, "record_chat_answer", fake_record_chat_answer)
    monkeypatch.setattr(feishu, "_schedule_feishu_answer_evaluation", lambda **_kwargs: None)

    asyncio.run(
        feishu._answer_feishu_message(
            question="question",
            sender_id="user-id",
            chat_id="chat-id",
            message_id="incoming-message-id",
            event_id="event-id",
            chat_type="p2p",
            dedupe_key="message:incoming-message-id",
        )
    )

    assert sent_messages == [("chat-id", "incoming-message-id", "step 1")]
    assert ("feedback-message-id", "step 2") in updates
    assert ("feedback-message-id", "step 3") in updates
    assert updates[-1] == ("feedback-message-id", "final answer")


def test_feishu_answer_schedules_evaluation_after_reply(monkeypatch) -> None:
    events: list[str] = []

    async def fake_send_message(
        *,
        chat_id: str | None,
        reply_to_message_id: str | None,
        markdown_text: str,
        **_kwargs,
    ) -> str:
        events.append(f"send:{markdown_text}")
        return "feedback-message-id"

    async def fake_update_message(message_id: str, markdown_text: str, **_kwargs) -> bool:
        events.append(f"update:{markdown_text}")
        return True

    async def fake_ask_knowledge_base(_request):
        events.append("query")
        return QueryResponse(question="question", answer="final answer")

    async def fake_create_chat_record(**_kwargs) -> None:
        return None

    async def fake_record_chat_answer(**_kwargs) -> None:
        events.append("record-answer")

    def fake_schedule_evaluation(**_kwargs) -> None:
        events.append("schedule-evaluation")

    def fail_if_evaluation_runs_inline(**_kwargs):
        raise AssertionError("evaluation should be scheduled after the Feishu reply")

    monkeypatch.setattr(feishu, "_load_feedback_texts", lambda: ["step 1"])
    monkeypatch.setattr(feishu, "_try_send_feishu_markdown_message", fake_send_message)
    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)
    monkeypatch.setattr(feishu, "ask_knowledge_base", fake_ask_knowledge_base)
    monkeypatch.setattr(feishu, "create_chat_record", fake_create_chat_record)
    monkeypatch.setattr(feishu, "record_chat_answer", fake_record_chat_answer)
    monkeypatch.setattr(feishu, "_schedule_feishu_answer_evaluation", fake_schedule_evaluation)
    monkeypatch.setattr(feishu, "evaluate_answer_fallback", fail_if_evaluation_runs_inline)

    asyncio.run(
        feishu._answer_feishu_message(
            question="question",
            sender_id="user-id",
            chat_id="chat-id",
            message_id="incoming-message-id",
            event_id="event-id",
            chat_type="p2p",
            dedupe_key="message:incoming-message-id",
        )
    )

    assert events == [
        "send:step 1",
        "query",
        "record-answer",
        "update:final answer",
        "schedule-evaluation",
    ]


def test_feishu_card_images_render_half_width_without_changing_text_width(monkeypatch) -> None:
    async def fake_upload_local_image(_raw_path: str, _token: str) -> str:
        return "img_v3_key"

    monkeypatch.setattr(feishu, "_upload_local_image", fake_upload_local_image)

    content = asyncio.run(
        feishu._build_feishu_card_content(
            "before\n<img>data/processing/demo/img/screenshot.png</img>\nafter",
            "tenant-token",
        )
    )

    card = json.loads(content)
    assert card["elements"][0] == {"tag": "markdown", "content": "before"}
    assert card["elements"][2] == {"tag": "markdown", "content": "after"}

    image_container = card["elements"][1]
    assert image_container["tag"] == "column_set"
    assert image_container["flex_mode"] == "bisect"
    assert [column["weight"] for column in image_container["columns"]] == [1, 1]

    image_column, spacer_column = image_container["columns"]
    assert spacer_column["elements"] == []
    assert image_column["elements"] == [
        {
            "tag": "img",
            "img_key": "img_v3_key",
            "alt": {
                "tag": "plain_text",
                "content": "screenshot.png",
            },
        }
    ]
