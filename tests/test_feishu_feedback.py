from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.api.v1 import feishu
from app.schemas.query import QueryResponse
from app.services.privacy import encrypt_chat_text


def test_extract_message_text_keeps_plain_text_messages() -> None:
    message = {
        "message_type": "text",
        "content": json.dumps({"text": " 工厂出厂含税价是多少？ "}, ensure_ascii=False),
    }

    assert feishu._extract_message_text(message) == "工厂出厂含税价是多少？"


def test_extract_message_text_supports_post_list_items() -> None:
    message = {
        "message_type": "post",
        "content": json.dumps(
            {
                "title": "",
                "content": [
                    [
                        {
                            "tag": "li",
                            "elements": [
                                {"tag": "text", "text": "工厂出厂含税价：32 ￥/KG"}
                            ],
                        }
                    ],
                    [
                        {
                            "tag": "li",
                            "elements": [
                                {"tag": "text", "text": "美元/人民币汇率：6.2"}
                            ],
                        }
                    ],
                    [
                        {
                            "tag": "li",
                            "elements": [
                                {"tag": "text", "text": "报价利润：5%"}
                            ],
                        }
                    ],
                ],
            },
            ensure_ascii=False,
        ),
    }

    assert feishu._extract_message_text(message) == (
        "- 工厂出厂含税价：32 ￥/KG\n"
        "- 美元/人民币汇率：6.2\n"
        "- 报价利润：5%"
    )


def test_extract_message_text_converts_post_links_to_markdown_and_skips_images() -> None:
    message = {
        "message_type": "post",
        "content": json.dumps(
            {
                "content": [
                    [
                        {"tag": "text", "text": "See "},
                        {"tag": "a", "text": "spec", "href": "https://example.test/spec"},
                    ],
                    [
                        {
                            "tag": "img",
                            "image_key": "img_v3_abc",
                            "alt": {"content": "quote screenshot"},
                        }
                    ],
                ],
            }
        ),
    }

    assert feishu._extract_message_text(message) == "See [spec](https://example.test/spec)"


def test_extract_message_text_rejects_image_messages() -> None:
    message = {
        "message_type": "image",
        "content": json.dumps({"image_key": "img_v3_xyz"}),
    }

    assert feishu._extract_message_text(message) == ""


def test_extract_message_text_rejects_unsupported_messages() -> None:
    message = {
        "message_type": "file",
        "content": json.dumps({"file_key": "file_v3_xyz"}),
    }

    assert feishu._extract_message_text(message) == ""


def test_unsupported_message_queues_notice_instead_of_silent_ignore(monkeypatch) -> None:
    queued_tasks: list[tuple[object, dict[str, object]]] = []

    class FakeBackgroundTasks:
        def add_task(self, func, **kwargs) -> None:
            queued_tasks.append((func, kwargs))

    class FakeRequest:
        client = "test-client"

        async def json(self) -> dict:
            return {
                "header": {
                    "event_type": "im.message.receive_v1",
                    "event_id": "event-unsupported",
                },
                "event": {
                    "sender": {"sender_id": {"open_id": "user-open-id"}},
                    "message": {
                        "message_id": "message-unsupported",
                        "chat_id": "chat-id",
                        "chat_type": "p2p",
                        "message_type": "image",
                        "content": json.dumps({"image_key": "img_v3_xyz"}),
                    },
                },
            }

    monkeypatch.setattr(feishu, "_seen_message_keys", {})

    result = asyncio.run(feishu.handle_feishu_events(FakeRequest(), FakeBackgroundTasks()))

    assert result == {
        "ok": True,
        "ignored": True,
        "reason": "unsupported_or_empty_message",
    }
    assert queued_tasks == [
        (
            feishu._send_unsupported_feishu_message_notice,
            {"chat_id": "chat-id", "message_id": "message-unsupported"},
        )
    ]


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

    async def fake_greeting(_question: str) -> str:
        return ""

    monkeypatch.setattr(feishu, "_feedback_interval_seconds", 0.05)
    monkeypatch.setattr(feishu, "_n8n_progress_polling_configured", lambda: False)
    monkeypatch.setattr(feishu, "_load_feedback_texts", lambda: ["step 1", "step 2", "step 3"])
    monkeypatch.setattr(feishu, "_generate_immediate_feedback_greeting", fake_greeting)
    monkeypatch.setattr(feishu, "_try_send_feishu_markdown_message", fake_send_message)
    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)
    monkeypatch.setattr(feishu, "ask_knowledge_base", fake_ask_knowledge_base)
    monkeypatch.setattr(feishu, "create_chat_record", fake_create_chat_record)
    monkeypatch.setattr(feishu, "record_chat_answer", fake_record_chat_answer)
    asyncio.run(
        feishu._answer_feishu_message(
            question="question",
            sender_id="user-id",
            sender_name="User Name",
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


def test_feishu_answer_does_not_schedule_evaluation_after_reply(monkeypatch) -> None:
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

    async def fake_greeting(_question: str) -> str:
        return ""

    monkeypatch.setattr(feishu, "_load_feedback_texts", lambda: ["step 1"])
    monkeypatch.setattr(feishu, "_generate_immediate_feedback_greeting", fake_greeting)
    monkeypatch.setattr(feishu, "_n8n_progress_polling_configured", lambda: False)
    monkeypatch.setattr(feishu, "_try_send_feishu_markdown_message", fake_send_message)
    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)
    monkeypatch.setattr(feishu, "ask_knowledge_base", fake_ask_knowledge_base)
    monkeypatch.setattr(feishu, "create_chat_record", fake_create_chat_record)
    monkeypatch.setattr(feishu, "record_chat_answer", fake_record_chat_answer)

    asyncio.run(
        feishu._answer_feishu_message(
            question="question",
            sender_id="user-id",
            sender_name="User Name",
            chat_id="chat-id",
            message_id="incoming-message-id",
            event_id="event-id",
            chat_type="p2p",
            dedupe_key="message:incoming-message-id",
        )
    )

    reply_event = "update:final answer" if "update:final answer" in events else "send:final answer"
    assert reply_event in events
    assert "record-answer" in events
    assert "schedule-evaluation" not in events


def test_feishu_initial_feedback_keeps_status_text_after_greeting(monkeypatch) -> None:
    async def fake_greeting(_question: str) -> str:
        return "收到，我先帮你看看这个问题。"

    monkeypatch.setattr(feishu, "_load_feedback_texts", lambda: ["正在检索知识库..."])
    monkeypatch.setattr(feishu, "_generate_immediate_feedback_greeting", fake_greeting)

    text = asyncio.run(feishu._build_initial_feedback_text("question"))

    assert text == "收到，我先帮你看看这个问题。\n\n正在检索知识库..."


def test_feishu_feedback_status_update_keeps_prefix(monkeypatch) -> None:
    updates: list[tuple[str, str]] = []

    async def fake_update_message(message_id: str, markdown_text: str, **_kwargs) -> bool:
        updates.append((message_id, markdown_text))
        return True

    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)
    state = feishu._FeishuFeedbackState(
        message_id="feedback-message-id",
        question="question",
        prefix_text="收到，我先帮你查一下。",
        status_text="🤔 正在理解问题...",
        update_lock=asyncio.Lock(),
    )

    asyncio.run(feishu._update_feishu_feedback_status(state, "📚 正在整理资料..."))

    assert updates == [
        ("feedback-message-id", "收到，我先帮你查一下。\n\n📚 正在整理资料...")
    ]


def test_feishu_feedback_update_keeps_cancel_footer_after_late_refresh(monkeypatch) -> None:
    updates: list[tuple[str | None, str | None]] = []

    async def fake_update_message(
        message_id: str,
        markdown_text: str,
        *,
        stop_cancel_id: str | None = None,
        canceled_text: str | None = None,
        **_kwargs,
    ) -> bool:
        updates.append((stop_cancel_id, canceled_text))
        return True

    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)
    state = feishu._FeishuFeedbackState(
        message_id="feedback-message-id",
        question="question",
        prefix_text="prefix",
        status_text="status",
        update_lock=asyncio.Lock(),
        cancel_id="cancel-1",
        canceled_text="cancelled",
    )

    asyncio.run(feishu._update_feishu_feedback_status(state, "late status"))

    assert updates == [(None, "cancelled")]


def test_feishu_cancelled_feedback_text_omits_status_line() -> None:
    state = feishu._FeishuFeedbackState(
        message_id="feedback-message-id",
        question="question",
        prefix_text="prefix",
        status_text="status should not show",
        update_lock=asyncio.Lock(),
        canceled_text="cancelled",
    )

    assert feishu._compose_cancelled_feedback_card_text(state) == "prefix"


def test_feishu_initial_feedback_skips_when_greeting_model_returns_null(monkeypatch) -> None:
    async def fake_greeting(_question: str) -> None:
        return None

    monkeypatch.setattr(feishu, "_load_feedback_texts", lambda: ["step 1"])
    monkeypatch.setattr(feishu, "_generate_immediate_feedback_greeting", fake_greeting)

    text = asyncio.run(feishu._build_initial_feedback_text("你好"))

    assert text is None


def test_immediate_feedback_cleaner_treats_null_as_skip_signal() -> None:
    assert feishu._clean_immediate_feedback_greeting("null") is None
    assert feishu._clean_immediate_feedback_greeting('"null"') is None
    assert feishu._clean_immediate_feedback_greeting("`null`") is None


def test_feishu_initial_feedback_does_not_block_query(monkeypatch) -> None:
    events: list[str] = []

    async def fake_send_message(
        *,
        chat_id: str | None,
        reply_to_message_id: str | None,
        markdown_text: str,
        **_kwargs,
    ) -> str:
        await asyncio.sleep(0.05)
        events.append(f"send:{markdown_text}")
        return "feedback-message-id"

    async def fake_update_message(message_id: str, markdown_text: str, **_kwargs) -> bool:
        events.append(f"update:{markdown_text}")
        return True

    async def fake_ask_knowledge_base(_request):
        events.append("query")
        await asyncio.sleep(0.15)
        return QueryResponse(question="question", answer="final answer")

    async def fake_create_chat_record(**_kwargs) -> None:
        return None

    async def fake_record_chat_answer(**_kwargs) -> None:
        return None

    async def fake_greeting(_question: str) -> str:
        await asyncio.sleep(0.05)
        return "收到，我先处理一下。"

    monkeypatch.setattr(feishu, "_load_feedback_texts", lambda: ["step 1"])
    monkeypatch.setattr(feishu, "_generate_immediate_feedback_greeting", fake_greeting)
    monkeypatch.setattr(feishu, "_n8n_progress_polling_configured", lambda: False)
    monkeypatch.setattr(feishu, "_try_send_feishu_markdown_message", fake_send_message)
    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)
    monkeypatch.setattr(feishu, "ask_knowledge_base", fake_ask_knowledge_base)
    monkeypatch.setattr(feishu, "create_chat_record", fake_create_chat_record)
    monkeypatch.setattr(feishu, "record_chat_answer", fake_record_chat_answer)
    asyncio.run(
        feishu._answer_feishu_message(
            question="question",
            sender_id="user-id",
            sender_name="User Name",
            chat_id="chat-id",
            message_id="incoming-message-id",
            event_id="event-id",
            chat_type="p2p",
            dedupe_key="message:incoming-message-id",
        )
    )

    assert events[0] == "query"
    assert "send:收到，我先处理一下。\n\nstep 1" in events
    assert events[-1] == "update:final answer"


def test_feishu_comfort_feedback_keeps_status_while_answer_is_slow(monkeypatch) -> None:
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
        await asyncio.sleep(0.14)
        return QueryResponse(question="question", answer="final answer")

    async def fake_create_chat_record(**_kwargs) -> None:
        return None

    async def fake_record_chat_answer(**_kwargs) -> None:
        return None

    async def fake_greeting(_question: str) -> str:
        return "收到，我先帮你看看。"

    async def fake_comfort_feedback_text(
        *,
        question: str,
        minute: int,
        status_text: str,
    ) -> str:
        assert question == "question"
        assert minute == 1
        assert status_text == "🤔 正在理解问题..."
        return "这个问题我还在对照资料，马上整理给你。"

    monkeypatch.setattr(feishu, "_feedback_interval_seconds", 10)
    monkeypatch.setattr(feishu, "_comfort_feedback_interval_seconds", 0.05)
    monkeypatch.setattr(feishu, "_n8n_progress_polling_configured", lambda: False)
    monkeypatch.setattr(feishu, "_load_feedback_texts", lambda: ["🤔 正在理解问题..."])
    monkeypatch.setattr(feishu, "_generate_immediate_feedback_greeting", fake_greeting)
    monkeypatch.setattr(feishu, "_generate_comfort_feedback_text", fake_comfort_feedback_text)
    monkeypatch.setattr(feishu, "_try_send_feishu_markdown_message", fake_send_message)
    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)
    monkeypatch.setattr(feishu, "ask_knowledge_base", fake_ask_knowledge_base)
    monkeypatch.setattr(feishu, "create_chat_record", fake_create_chat_record)
    monkeypatch.setattr(feishu, "record_chat_answer", fake_record_chat_answer)
    asyncio.run(
        feishu._answer_feishu_message(
            question="question",
            sender_id="user-id",
            sender_name="User Name",
            chat_id="chat-id",
            message_id="incoming-message-id",
            event_id="event-id",
            chat_type="p2p",
            dedupe_key="message:incoming-message-id",
        )
    )

    assert "send:收到，我先帮你看看。\n\n🤔 正在理解问题..." in events
    comfort_updates = [
        event
        for event in events
        if event
        == "update:这个问题我还在对照资料，马上整理给你。\n\n🤔 正在理解问题..."
    ]
    assert comfort_updates
    assert events[-1] == "update:final answer"


def test_feishu_null_greeting_does_not_send_feedback_or_block_query(monkeypatch) -> None:
    events: list[str] = []

    async def fake_send_message(
        *,
        chat_id: str | None,
        reply_to_message_id: str | None,
        markdown_text: str,
        **_kwargs,
    ) -> str:
        events.append(f"send:{markdown_text}")
        return "sent-message-id"

    async def fake_update_message(message_id: str, markdown_text: str, **_kwargs) -> bool:
        events.append(f"update:{markdown_text}")
        return True

    async def fake_ask_knowledge_base(_request):
        events.append("query")
        return QueryResponse(question="你好", answer="你好，有什么可以帮你？")

    async def fake_create_chat_record(**_kwargs) -> None:
        return None

    async def fake_record_chat_answer(**_kwargs) -> None:
        events.append("record-answer")

    async def fake_greeting(_question: str) -> None:
        events.append("greeting-null")
        return None

    monkeypatch.setattr(feishu, "_load_feedback_texts", lambda: ["step 1"])
    monkeypatch.setattr(feishu, "_generate_immediate_feedback_greeting", fake_greeting)
    monkeypatch.setattr(feishu, "_n8n_progress_polling_configured", lambda: False)
    monkeypatch.setattr(feishu, "_try_send_feishu_markdown_message", fake_send_message)
    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)
    monkeypatch.setattr(feishu, "ask_knowledge_base", fake_ask_knowledge_base)
    monkeypatch.setattr(feishu, "create_chat_record", fake_create_chat_record)
    monkeypatch.setattr(feishu, "record_chat_answer", fake_record_chat_answer)

    asyncio.run(
        feishu._answer_feishu_message(
            question="你好",
            sender_id="user-id",
            sender_name="User Name",
            chat_id="chat-id",
            message_id="incoming-message-id",
            event_id="event-id",
            chat_type="p2p",
            dedupe_key="message:incoming-message-id",
        )
    )

    assert "send:step 1" not in events
    assert "update:step 1" not in events
    assert "query" in events
    assert "send:你好，有什么可以帮你？" in events


def test_feishu_transient_progress_answer_keeps_initial_feedback(monkeypatch) -> None:
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
        return QueryResponse(question="question", answer="正在整理资料...")

    async def fake_create_chat_record(**_kwargs) -> None:
        return None

    async def fail_record_chat_answer(**_kwargs) -> None:
        raise AssertionError("transient progress text should not be recorded as final answer")

    async def fake_greeting(_question: str) -> str:
        await asyncio.sleep(0.01)
        return "收到，我先帮你查一下。"

    monkeypatch.setattr(feishu, "_load_feedback_texts", lambda: ["正在整理资料..."])
    monkeypatch.setattr(feishu, "_generate_immediate_feedback_greeting", fake_greeting)
    monkeypatch.setattr(feishu, "_n8n_progress_polling_configured", lambda: False)
    monkeypatch.setattr(feishu, "_try_send_feishu_markdown_message", fake_send_message)
    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)
    monkeypatch.setattr(feishu, "ask_knowledge_base", fake_ask_knowledge_base)
    monkeypatch.setattr(feishu, "create_chat_record", fake_create_chat_record)
    monkeypatch.setattr(feishu, "record_chat_answer", fail_record_chat_answer)
    asyncio.run(
        feishu._answer_feishu_message(
            question="question",
            sender_id="user-id",
            sender_name="User Name",
            chat_id="chat-id",
            message_id="incoming-message-id",
            event_id="event-id",
            chat_type="p2p",
            dedupe_key="message:incoming-message-id",
        )
    )

    assert events[0] == "query"
    assert "send:收到，我先帮你查一下。\n\n正在整理资料..." in events
    assert "update:正在整理资料..." not in events


def test_immediate_feedback_uses_deepseek_enable_thinking_off(monkeypatch) -> None:
    calls: list[dict[str, object] | None] = []

    class FakeLLMClient:
        def __init__(self, _settings) -> None:
            pass

        def chat(self, _messages, **kwargs) -> str:
            calls.append(kwargs.get("extra_body"))
            return "收到，我马上处理。"

    monkeypatch.setattr(
        feishu,
        "settings",
        SimpleNamespace(
            siliconflow_api_key="test-key",
            siliconflow_base_url="https://example.test/v1",
            immediate_feedback_model="deepseek-ai/DeepSeek-V4-Flash",
            immediate_feedback_timeout=3,
            immediate_feedback_connect_timeout=1,
            immediate_feedback_max_tokens=80,
            immediate_feedback_enable_thinking="off",
        ),
    )
    monkeypatch.setattr(feishu, "LLMClient", FakeLLMClient)

    reply = feishu._generate_immediate_feedback_greeting_sync("question")

    assert reply == "收到，我马上处理。"
    assert calls == [{"enable_thinking": False}]


def test_immediate_feedback_model_failure_uses_local_greeting(monkeypatch) -> None:
    async def fake_greeting(_question: str) -> str:
        return "收到，我先帮你查一下相关资料，再整理成好读的答复。"

    monkeypatch.setattr(feishu, "_load_feedback_texts", lambda: ["🤔 正在理解问题..."])
    monkeypatch.setattr(feishu, "_generate_immediate_feedback_greeting", fake_greeting)

    text = asyncio.run(feishu._build_initial_feedback_text("如何查询订单谈判注意事项？"))

    assert text == "收到，我先帮你查一下相关资料，再整理成好读的答复。\n\n🤔 正在理解问题..."


def test_feishu_card_images_render_as_half_width_column_sets(monkeypatch) -> None:
    async def fake_upload_local_image(_raw_path: str, _token: str) -> str:
        return "img_v3_key"

    monkeypatch.setattr(feishu, "_upload_local_image", fake_upload_local_image)
    monkeypatch.setattr(feishu, "_get_local_image_size", lambda _raw_path: (1200, 800))

    content = asyncio.run(
        feishu._build_feishu_card_content(
            "before\n<img>data/processing/demo/img/screenshot.png</img>\nafter",
            "tenant-token",
        )
    )

    card = json.loads(content)
    assert card["schema"] == "2.0"
    elements = card["body"]["elements"]
    assert elements[0] == {
        "tag": "markdown",
        "content": "before",
        "text_align": "left",
        "text_size": "normal_v2",
    }
    image_container = elements[1]
    assert image_container["tag"] == "column_set"
    assert image_container["flex_mode"] == "bisect"
    assert len(image_container["columns"]) == 2
    assert image_container["columns"][0]["elements"] == [
        {
            "tag": "img",
            "img_key": "img_v3_key",
            "alt": {
                "tag": "plain_text",
                "content": "screenshot.png",
            },
        }
    ]
    assert image_container["columns"][1]["elements"] == []
    assert elements[2] == {
        "tag": "markdown",
        "content": "after",
        "text_align": "left",
        "text_size": "normal_v2",
    }


def test_feishu_card_renders_similar_consecutive_images_side_by_side(monkeypatch) -> None:
    async def fake_upload_local_image(raw_path: str, _token: str) -> str:
        return f"key_{raw_path[-5]}"

    image_sizes = {
        "data/processing/demo/img/a.png": (1000, 800),
        "data/processing/demo/img/b.png": (980, 780),
    }

    monkeypatch.setattr(feishu, "_upload_local_image", fake_upload_local_image)
    monkeypatch.setattr(feishu, "_get_local_image_size", lambda raw_path: image_sizes[raw_path])

    content = asyncio.run(
        feishu._build_feishu_card_content(
            "<img>data/processing/demo/img/a.png</img>\n"
            "<img>data/processing/demo/img/b.png</img>",
            "tenant-token",
        )
    )

    card = json.loads(content)
    elements = card["body"]["elements"]
    assert len(elements) == 1
    assert elements[0]["tag"] == "column_set"
    assert [column["elements"][0]["img_key"] for column in elements[0]["columns"]] == [
        "key_a",
        "key_b",
    ]


def test_feishu_card_renders_different_consecutive_images_vertically(monkeypatch) -> None:
    async def fake_upload_local_image(raw_path: str, _token: str) -> str:
        return f"key_{raw_path[-5]}"

    image_sizes = {
        "data/processing/demo/img/a.png": (1200, 900),
        "data/processing/demo/img/b.png": (240, 180),
    }

    monkeypatch.setattr(feishu, "_upload_local_image", fake_upload_local_image)
    monkeypatch.setattr(feishu, "_get_local_image_size", lambda raw_path: image_sizes[raw_path])

    content = asyncio.run(
        feishu._build_feishu_card_content(
            "<img>data/processing/demo/img/a.png</img>\n"
            "<img>data/processing/demo/img/b.png</img>",
            "tenant-token",
        )
    )

    card = json.loads(content)
    elements = card["body"]["elements"]
    assert [element["tag"] for element in elements] == ["column_set", "column_set"]
    assert elements[0]["columns"][0]["elements"][0]["img_key"] == "key_a"
    assert elements[0]["columns"][1]["elements"] == []
    assert elements[1]["columns"][0]["elements"][0]["img_key"] == "key_b"
    assert elements[1]["columns"][1]["elements"] == []


def test_feishu_card_preserves_markdown_for_json2_renderer() -> None:
    markdown = (
        "### 标题\n\n"
        "> 引用内容\n\n"
        "| 列一 | 列二 |\n"
        "| --- | --- |\n"
        "| A | B |\n\n"
        "---\n"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert card["schema"] == "2.0"
    assert card["body"]["elements"] == [
        {
            "tag": "markdown",
            "content": markdown.strip(),
            "text_align": "left",
            "text_size": "normal_v2",
        }
    ]


def test_feishu_card_renders_latex_formula_as_readable_text() -> None:
    markdown = (
        r"\[\text{FOB USD/kg}=\frac{3.590+0.057+0.108}{0.99}=3.79\ \text{USD/kg}\]"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert (
        card["body"]["elements"][0]["content"]
        == "FOB USD/kg = (3.590 + 0.057 + 0.108) / (0.99) = 3.79 USD/kg"
    )


def test_feishu_card_renders_explicit_latex_block_without_latex_commands() -> None:
    markdown = r"\[x=1\]"

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert card["body"]["elements"][0]["content"] == "x = 1"


def test_feishu_card_renders_bare_bracket_latex_formula() -> None:
    markdown = (
        r"Result: [\text{FOB USD/kg}=\frac{3.590+0.057+0.108}{0.99}=3.79\ \text{USD/kg}]"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert (
        card["body"]["elements"][0]["content"]
        == "Result: FOB USD/kg = (3.590 + 0.057 + 0.108) / (0.99) = 3.79 USD/kg"
    )


def test_feishu_card_renders_bare_bracket_latex_formula_with_times() -> None:
    markdown = r"[\frac{26.5}{7.10\times 0.87\times 0.92} = 4.663\ \text{USD/kg}]"

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert (
        card["body"]["elements"][0]["content"]
        == "(26.5) / (7.10 x 0.87 x 0.92) = 4.663 USD/kg"
    )


def test_feishu_card_renders_standalone_multiline_bracket_formula() -> None:
    markdown = (
        "[\n"
        r"\text{加利润后 USD/kg}=\frac{23.45}{7.10\times 0.92}\approx 3.589\ \text{USD/kg}"
        "\n]"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert (
        card["body"]["elements"][0]["content"]
        == "加利润后 USD/kg = (23.45) / (7.10 x 0.92) ~= 3.589 USD/kg"
    )


def test_feishu_card_renders_latex_text_formatting_commands() -> None:
    markdown = r"[\text{FOB USD/kg}=\frac{3.589+0.0535+0.1080}{0.99}\approx \mathbf{3.79}\ \text{USD/kg}]"

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert (
        card["body"]["elements"][0]["content"]
        == "FOB USD/kg = (3.589 + 0.0535 + 0.1080) / (0.99) ~= 3.79 USD/kg"
    )


def test_feishu_card_keeps_non_formula_multiline_bracket_text() -> None:
    markdown = "[\nplain text\n]"

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert card["body"]["elements"][0]["content"] == markdown


def test_feishu_card_keeps_latex_inside_code_blocks() -> None:
    markdown = "```\n" + r"\[\frac{1}{2}\]" + "\n```"

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert card["body"]["elements"][0]["content"] == markdown


def test_feishu_card_rewrites_reference_links_and_appends_source_panel() -> None:
    expected_url = feishu._format_markdown_link_url(
        "data/raw/结构化word文档/造船行业.docx"
    )
    markdown = (
        "结论内容。"
        "[片段1,造船行业发展情况说明](data/raw/结构化word文档/造船行业.docx)"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    elements = card["body"]["elements"]
    assert elements[0] == {
        "tag": "markdown",
        "content": f"结论内容。[[1]]({expected_url})",
        "text_align": "left",
        "text_size": "normal_v2",
    }
    assert elements[1] == {
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {"title": {"tag": "plain_text", "content": "知识来源"}},
        "elements": [
            {
                "tag": "markdown",
                "content": f"[[1]]({expected_url}) 片段1,造船行业发展情况说明",
                "text_align": "left",
                "text_size": "normal_v2",
            }
        ],
    }


def test_feishu_card_rewrites_reference_tags_and_appends_source_panel() -> None:
    expected_url = feishu._build_document_download_url("data/raw/a.docx")
    markdown = "Answer from KB<reference>data/raw/a.docx</Reference>"

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    elements = card["body"]["elements"]
    assert elements[0]["content"] == f"Answer from KB[[1]]({expected_url})"
    assert elements[1]["elements"][0]["content"] == f"[[1]]({expected_url}) a.docx"


def test_feishu_card_deduplicates_reference_tags() -> None:
    expected_url = feishu._build_document_download_url("data/raw/a.docx")
    markdown = (
        "First<reference>data/raw/a.docx</reference>\n"
        "Second<Reference>data\\raw\\a.docx</Reference>"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    elements = card["body"]["elements"]
    assert elements[0]["content"] == f"First[[1]]({expected_url})\nSecond[[1]]({expected_url})"
    assert elements[1]["elements"][0]["content"] == f"[[1]]({expected_url}) a.docx"


def test_feishu_card_joins_reference_tags_split_across_lines() -> None:
    url_a = feishu._build_document_download_url("data/raw/a.docx")
    url_b = feishu._build_document_download_url("data/raw/b.docx")
    url_c = feishu._build_document_download_url("data/raw/c.docx")
    markdown = (
        "结论内容。<reference>data/raw/a.docx</reference>\n"
        "<Reference>data/raw/b.docx</Reference>\n"
        "<reference>data/raw/c.docx</reference>"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    main_content = card["body"]["elements"][0]["content"]
    assert main_content == f"结论内容。[[1]]({url_a})[[2]]({url_b})[[3]]({url_c})"
    assert f"[[1]]({url_a})\n[[2]]({url_b})" not in main_content


def test_feishu_card_joins_existing_short_references_split_across_lines() -> None:
    url_a = feishu._build_document_download_url("data/raw/a.docx")
    url_b = feishu._build_document_download_url("data/raw/b.docx")
    markdown = f"结论内容。[[1]]({url_a})\n[[2]]({url_b})"

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert card["body"]["elements"][0]["content"] == f"结论内容。[[1]]({url_a})[[2]]({url_b})"


def test_feishu_card_keeps_reference_tags_inside_code_blocks() -> None:
    markdown = "```\n<reference>data/raw/a.docx</reference>\n```"

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert card["body"]["elements"][0]["content"] == markdown
    assert len(card["body"]["elements"]) == 1


def test_feishu_card_rewrites_legacy_reference_title_links() -> None:
    expected_url = feishu._format_markdown_link_url(
        "data/raw/结构化word文档/造船行业.docx"
    )
    markdown = (
        "结论内容。"
        '[片段1](data/raw/结构化word文档/造船行业.docx "片段1,造船行业发展情况说明")'
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    elements = card["body"]["elements"]
    assert elements[0]["content"] == f"结论内容。[[1]]({expected_url})"
    assert (
        elements[1]["elements"][0]["content"]
        == f"[[1]]({expected_url}) 片段1,造船行业发展情况说明"
    )


def test_feishu_card_rewrites_reference_links_with_windows_paths_and_spaces() -> None:
    markdown = (
        "提醒客户：如果有工厂承诺春节期间还能快速交货，很可能是为了接单而过分承诺，"
        "后期反而容易出问题。"
        '[片段13](data\\raw\\一般类、文本居多的word文档\\销售工具包 - 订单谈判.docx '
        '"片段13,客户无法理解为什么春节假期的交货期要延长这么久怎么办")'
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    elements = card["body"]["elements"]
    expected_url = feishu._format_markdown_link_url(
        "data\\raw\\一般类、文本居多的word文档\\销售工具包 - 订单谈判.docx"
    )
    assert elements[0]["content"].endswith(f"[[1]]({expected_url})")
    assert (
        elements[1]["elements"][0]["content"]
        == f"[[1]]({expected_url}) 片段13,客户无法理解为什么春节假期的交货期要延长这么久怎么办"
    )


def test_feishu_card_deduplicates_reference_sources_and_strips_model_source_section() -> None:
    expected_url = feishu._build_document_download_url("data/raw/a.docx")
    markdown = (
        "第一段。[片段7,来源甲](data/raw/a.docx)\n"
        "第二段。[片段2,重复来源](data/raw/a.docx)\n\n"
        "---\n"
        "知识来源：\n"
        "[片段7,来源甲](data/raw/a.docx)"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    elements = card["body"]["elements"]
    assert elements[0]["content"] == f"第一段。[[1]]({expected_url})\n第二段。[[1]]({expected_url})"
    assert elements[1]["elements"][0]["content"] == f"[[1]]({expected_url}) 片段7,来源甲"


def test_feishu_card_deduplicates_adjacent_repeated_reference_links() -> None:
    expected_url = feishu._build_document_download_url("data/raw/a.docx")
    markdown = (
        "结论内容。"
        "[片段7,来源甲](data/raw/a.docx)"
        "[片段2,重复来源](data/raw/a.docx)"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    elements = card["body"]["elements"]
    assert elements[0]["content"] == f"结论内容。[[1]]({expected_url})"
    assert elements[1]["elements"][0]["content"] == f"[[1]]({expected_url}) 片段7,来源甲"


def test_feishu_card_deduplicates_adjacent_existing_short_references() -> None:
    existing_url = feishu._build_document_download_url("data/raw/a.docx")
    markdown = f"结论内容。[[1]]({existing_url})[[1]]({existing_url})"

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert card["body"]["elements"][0]["content"] == f"结论内容。[[1]]({existing_url})"


def test_feishu_card_keeps_regular_markdown_links_and_code_blocks() -> None:
    markdown = (
        "[普通链接](https://example.com)\n"
        "```\n"
        "[片段1,不要处理](demo.docx)\n"
        "```"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert card["body"]["elements"][0]["content"] == markdown


def test_format_markdown_link_url_uses_lark_mapping_by_filename(monkeypatch) -> None:
    monkeypatch.setattr(
        feishu,
        "_local_to_lark_mapping_cache",
        {"demofile.docx": "https://example.feishu.cn/wiki/demo"},
    )

    assert (
        feishu._format_markdown_link_url("data\\raw\\folder\\Demo File.docx")
        == "https://example.feishu.cn/wiki/demo"
    )


def test_format_markdown_link_url_extracts_filename_from_download_url(monkeypatch) -> None:
    monkeypatch.setattr(
        feishu,
        "_local_to_lark_mapping_cache",
        {"demofile.docx": "https://example.feishu.cn/wiki/demo"},
    )
    raw_url = feishu._build_document_download_url("data/raw/folder/Demo File.docx")

    assert feishu._format_markdown_link_url(raw_url) == "https://example.feishu.cn/wiki/demo"


def test_format_markdown_link_url_falls_back_to_download_url(monkeypatch) -> None:
    monkeypatch.setattr(feishu, "_local_to_lark_mapping_cache", {})
    raw_path = "data/raw/folder/missing.docx"

    assert feishu._format_markdown_link_url(raw_path) == feishu._build_document_download_url(raw_path)


def test_format_markdown_link_url_keeps_existing_lark_url() -> None:
    raw_url = "https://example.feishu.cn/wiki/demo"

    assert feishu._format_markdown_link_url(raw_url) == raw_url


def test_n8n_progress_stage_recognizes_workflow_trigger_by_id() -> None:
    execution = {
        "workflowData": {
            "nodes": [
                {
                    "id": "e0e954af-c14c-47d1-917a-bdbc69eba580",
                    "name": "renamed trigger node",
                }
            ]
        },
        "data": {
            "resultData": {
                "runData": {
                    "renamed trigger node": [
                        {"startTime": "2026-06-15T01:00:00.000Z"}
                    ]
                }
            }
        },
    }

    assert feishu._extract_n8n_progress_stage(execution) == "understanding"


def test_n8n_progress_stage_advances_after_switch_finishes() -> None:
    execution = {
        "workflowData": {
            "nodes": [
                {
                    "id": "bdb2ff5c-c9b8-40f1-b623-dc8b912c0600",
                    "name": "renamed switch node",
                }
            ]
        },
        "data": {
            "resultData": {
                "runData": {
                    "renamed switch node": [
                        {"startTime": "2026-06-15T01:00:00.000Z"}
                    ]
                }
            }
        },
    }

    assert feishu._extract_n8n_progress_stage(execution) == "retrieving"


def test_n8n_progress_stage_advances_after_retrieval_node_finishes() -> None:
    execution = {
        "workflowData": {
            "nodes": [
                {
                    "id": "53fb4814-a531-4422-b6ba-f38c3db5a9a4",
                    "name": "renamed retrieval node",
                }
            ]
        },
        "data": {
            "resultData": {
                "runData": {
                    "renamed retrieval node": [
                        {"startTime": "2026-06-15T01:00:00.000Z"}
                    ]
                }
            }
        },
    }

    assert feishu._extract_n8n_progress_stage(execution) == "reranking"


def test_n8n_progress_stage_advances_after_context_formatting_finishes() -> None:
    execution = {
        "workflowData": {
            "nodes": [
                {
                    "id": "6113024b-7803-4ce2-930a-c893ff1ff7fd",
                    "name": "renamed context formatter",
                }
            ]
        },
        "data": {
            "resultData": {
                "runData": {
                    "renamed context formatter": [
                        {"startTime": "2026-06-15T01:00:00.000Z"}
                    ]
                }
            }
        },
    }

    assert feishu._extract_n8n_progress_stage(execution) == "generating"


def test_n8n_progress_stage_ignores_fast_chat_node() -> None:
    execution = {
        "data": {
            "resultData": {
                "runData": {
                    "无关闲聊": [
                        {"startTime": "2026-06-15T01:00:00.000Z"}
                    ]
                }
            }
        },
    }

    assert feishu._extract_n8n_progress_stage(execution) is None


def test_n8n_progress_stage_ignores_rewrite_node_id() -> None:
    execution = {
        "workflowData": {
            "nodes": [
                {
                    "id": "54aad033-e2d6-4b2a-aa73-027f9fc839ce",
                    "name": "renamed rewrite node",
                }
            ]
        },
        "data": {
            "resultData": {
                "runData": {
                    "renamed rewrite node": [
                        {"startTime": "2026-06-15T01:00:00.000Z"}
                    ]
                }
            }
        },
    }

    assert feishu._extract_n8n_progress_stage(execution) is None


def test_n8n_progress_stage_ignores_retrieval_child_workflow_nodes() -> None:
    execution = {
        "workflowData": {
            "nodes": [
                {
                    "id": "db342e62-4de8-486a-9443-ddfafc679b77",
                    "name": "renamed child code node",
                }
            ]
        },
        "data": {
            "resultData": {
                "runData": {
                    "renamed child code node": [
                        {"startTime": "2026-06-15T01:00:00.000Z"}
                    ]
                }
            }
        },
    }

    assert feishu._extract_n8n_progress_stage(execution) is None


def test_optimistic_n8n_progress_stage_advances_by_elapsed_time() -> None:
    assert feishu._optimistic_n8n_progress_stage(elapsed_seconds=1.0) is None
    assert feishu._optimistic_n8n_progress_stage(elapsed_seconds=1.5) == "retrieving"
    assert feishu._optimistic_n8n_progress_stage(elapsed_seconds=5.0) == "reranking"
    assert feishu._optimistic_n8n_progress_stage(elapsed_seconds=10.0) == "generating"


def test_latest_n8n_progress_stage_prefers_later_stage() -> None:
    assert feishu._latest_n8n_progress_stage("understanding", "reranking") == "reranking"
    assert feishu._latest_n8n_progress_stage("generating", "retrieving") == "generating"
    assert feishu._latest_n8n_progress_stage(None, "retrieving") == "retrieving"


def test_n8n_progress_stage_returns_latest_mapped_stage() -> None:
    execution = {
        "workflowData": {
            "nodes": [
                {
                    "id": "e0e954af-c14c-47d1-917a-bdbc69eba580",
                    "name": "renamed trigger node",
                },
                {
                    "id": "bdb2ff5c-c9b8-40f1-b623-dc8b912c0600",
                    "name": "renamed switch node",
                },
                {
                    "id": "53fb4814-a531-4422-b6ba-f38c3db5a9a4",
                    "name": "renamed retrieval entry",
                },
                {
                    "id": "6113024b-7803-4ce2-930a-c893ff1ff7fd",
                    "name": "renamed context formatter",
                },
            ]
        },
        "data": {
            "resultData": {
                "runData": {
                    "renamed trigger node": [
                        {"startTime": "2026-06-15T01:00:00.000Z"}
                    ],
                    "renamed switch node": [
                        {"startTime": "2026-06-15T01:00:01.000Z"}
                    ],
                    "renamed retrieval entry": [
                        {"startTime": "2026-06-15T01:00:02.000Z"}
                    ],
                    "renamed context formatter": [
                        {"startTime": "2026-06-15T01:00:04.000Z"}
                    ],
                }
            }
        },
    }

    assert feishu._extract_n8n_progress_stage(execution) == "generating"


def test_feishu_waiting_card_adds_stop_button() -> None:
    content = asyncio.run(
        feishu._build_feishu_card_content(
            "正在处理...",
            "tenant-token",
            stop_cancel_id="cancel-1",
        )
    )

    card = json.loads(content)
    button = card["body"]["elements"][-1]
    assert button["tag"] == "button"
    assert button["type"] == "danger"
    assert button["behaviors"] == [
        {
            "type": "callback",
            "value": {
                "action": "stop_feishu_answer",
                "cancel_id": "cancel-1",
            },
        }
    ]


def test_feishu_waiting_card_does_not_use_unsupported_action_container() -> None:
    content = asyncio.run(
        feishu._build_feishu_card_content(
            "姝ｅ湪澶勭悊...",
            "tenant-token",
            stop_cancel_id="cancel-1",
        )
    )

    card = json.loads(content)
    assert all(element.get("tag") != "action" for element in card["body"]["elements"])


def test_feishu_final_answer_card_adds_radio_feedback_buttons() -> None:
    feedback_url = feishu._get_answer_feedback_form_url()
    content = asyncio.run(
        feishu._build_feishu_card_content(
            "final answer",
            "tenant-token",
            answer_feedback_id="feedback-1",
        )
    )

    card = json.loads(content)
    elements = card["body"]["elements"]
    feedback_panel = elements[-1]
    assert feedback_panel["tag"] == "collapsible_panel"
    assert feedback_panel["expanded"] is False
    assert feedback_panel["header"] == {
        "title": {"tag": "plain_text", "content": "反馈"}
    }

    feedback_elements = feedback_panel["elements"]
    button_row = feedback_elements[1]
    assert button_row["tag"] == "column_set"

    helpful_button = button_row["columns"][0]["elements"][0]
    issue_button = button_row["columns"][1]["elements"][0]
    assert helpful_button["text"]["content"] == "👍 有帮助"
    assert helpful_button["behaviors"] == [
        {
            "type": "callback",
            "value": {
                "action": "answer_feedback",
                "feedback_id": "feedback-1",
                "choice": "helpful",
            },
        }
    ]
    assert issue_button["text"]["content"] == "👎 有问题，去反馈"
    assert issue_button["behaviors"][0]["value"] == {
        "action": "answer_feedback",
        "feedback_id": "feedback-1",
        "choice": "issue",
    }
    assert issue_button["behaviors"][1]["type"] == "open_url"
    assert issue_button["behaviors"][1]["default_url"] == feedback_url
    assert f"[去反馈]({feedback_url})" in feedback_elements[-1]["content"]


def test_feishu_final_answer_feedback_selected_locks_other_choice() -> None:
    content = asyncio.run(
        feishu._build_feishu_card_content(
            "final answer",
            "tenant-token",
            answer_feedback_id="feedback-1",
            answer_feedback_selected="helpful",
        )
    )

    card = json.loads(content)
    feedback_panel = card["body"]["elements"][-1]
    button_row = feedback_panel["elements"][1]
    helpful_button = button_row["columns"][0]["elements"][0]
    issue_button = button_row["columns"][1]["elements"][0]
    assert helpful_button["type"] == "primary"
    assert helpful_button["text"]["content"] == "👍 已标记有帮助"
    assert "disabled" not in helpful_button
    assert issue_button["disabled"] is True


def test_feishu_answer_feedback_action_keeps_first_choice(monkeypatch) -> None:
    async def fake_get_token() -> str:
        return "tenant-token"

    async def run_case() -> tuple[dict, dict]:
        feishu._feishu_answer_feedback_states["feedback-1"] = feishu._FeishuAnswerFeedbackState(
            feedback_id="feedback-1",
            answer="final answer",
            selected=None,
            created_at=feishu.time.monotonic(),
        )
        first = await feishu._handle_feishu_card_action(
            {
                "event": {
                    "action": {
                        "value": {
                            "action": "answer_feedback",
                            "feedback_id": "feedback-1",
                            "choice": "issue",
                        }
                    }
                }
            }
        )
        second = await feishu._handle_feishu_card_action(
            {
                "event": {
                    "action": {
                        "value": {
                            "action": "answer_feedback",
                            "feedback_id": "feedback-1",
                            "choice": "helpful",
                        }
                    }
                }
            }
        )
        return first, second

    monkeypatch.setattr(feishu, "_get_tenant_access_token", fake_get_token)

    try:
        first_result, second_result = asyncio.run(run_case())
    finally:
        feishu._feishu_answer_feedback_states.pop("feedback-1", None)

    assert first_result["selected"] == "issue"
    assert second_result["selected"] == "issue"
    assert second_result["toast"]["type"] == "info"
    first_elements = first_result["card"]["data"]["body"]["elements"]
    first_feedback_panel = first_elements[-1]
    first_button_row = first_feedback_panel["elements"][1]
    assert first_button_row["columns"][1]["elements"][0]["type"] == "danger"


def test_feishu_answer_feedback_action_recovers_state_from_database(monkeypatch) -> None:
    async def fake_get_token() -> str:
        return "tenant-token"

    def fake_get_state(feedback_id: str) -> dict:
        assert feedback_id == "feedback-db"
        return {
            "feedback_id": feedback_id,
            "answer": encrypt_chat_text("final answer"),
            "selected": None,
            "create_time": datetime.now(timezone.utc),
        }

    def fake_update_selection(*, feedback_id: str, selected: str) -> dict:
        return {
            "feedback_id": feedback_id,
            "answer": encrypt_chat_text("final answer"),
            "selected": selected,
            "create_time": datetime.now(timezone.utc),
        }

    async def run_case() -> dict:
        feishu._feishu_answer_feedback_states.pop("feedback-db", None)
        return await feishu._handle_feishu_card_action(
            {
                "event": {
                    "action": {
                        "value": {
                            "action": "answer_feedback",
                            "feedback_id": "feedback-db",
                            "choice": "helpful",
                        }
                    }
                }
            }
        )

    monkeypatch.setattr(feishu, "_get_tenant_access_token", fake_get_token)
    monkeypatch.setattr(feishu, "delete_expired_feishu_answer_feedback_states", lambda **_kwargs: 0)
    monkeypatch.setattr(feishu, "get_feishu_answer_feedback_state", fake_get_state)
    monkeypatch.setattr(feishu, "update_feishu_answer_feedback_selection", fake_update_selection)

    try:
        result = asyncio.run(run_case())
    finally:
        feishu._feishu_answer_feedback_states.pop("feedback-db", None)

    assert result["selected"] == "helpful"
    feedback_panel = result["card"]["data"]["body"]["elements"][-1]
    button_row = feedback_panel["elements"][1]
    assert button_row["columns"][0]["elements"][0]["type"] == "primary"


def test_feishu_card_content_error_detection() -> None:
    exc = feishu.HTTPException(
        status_code=502,
        detail="Failed to create card content; unsupported tag action",
    )

    assert feishu._is_feishu_card_content_error(exc) is True


def test_feishu_card_action_value_can_be_extracted_from_button_behavior_payload() -> None:
    value = feishu._extract_feishu_card_action_value(
        {
            "header": {"event_type": "card.action.trigger"},
            "event": {
                "action": {
                    "value": {
                        "action": "stop_feishu_answer",
                        "cancel_id": "cancel-1",
                    }
                }
            },
        }
    )

    assert value == {
        "action": "stop_feishu_answer",
        "cancel_id": "cancel-1",
    }


def test_format_feishu_cancel_elapsed_omits_minutes_under_one_minute() -> None:
    assert feishu._format_feishu_cancel_elapsed(started_at=10.0, canceled_at=13.2) == "您在3秒后取消回答"
    assert feishu._format_feishu_cancel_elapsed(started_at=10.0, canceled_at=75.0) == "您在1分5秒后取消回答"


def test_feishu_cancelled_card_shows_footer_without_stop_button() -> None:
    content = asyncio.run(
        feishu._build_feishu_card_content(
            "正在处理...",
            "tenant-token",
            stop_cancel_id="cancel-1",
            canceled_text="您在3秒后取消回答",
        )
    )

    card = json.loads(content)
    elements = card["body"]["elements"]
    assert elements[-2] == {"tag": "hr"}
    assert "您在3秒后取消回答" in elements[-1]["content"]
    assert all(element.get("tag") != "action" for element in elements)


def test_feishu_cancel_action_suppresses_final_answer(monkeypatch) -> None:
    updates: list[tuple[str, str, str | None]] = []
    sent_cancel_ids: list[str | None] = []
    query_started = asyncio.Event()

    async def fake_send_message(
        *,
        chat_id: str | None,
        reply_to_message_id: str | None,
        markdown_text: str,
        stop_cancel_id: str | None = None,
        **_kwargs,
    ) -> str:
        sent_cancel_ids.append(stop_cancel_id)
        return "feedback-message-id"

    async def fake_update_message(
        message_id: str,
        markdown_text: str,
        *,
        canceled_text: str | None = None,
        **_kwargs,
    ) -> bool:
        updates.append((message_id, markdown_text, canceled_text))
        return True

    async def fake_ask_knowledge_base(_request):
        query_started.set()
        await asyncio.sleep(30)
        return QueryResponse(question="question", answer="final answer")

    async def fake_create_chat_record(**_kwargs) -> None:
        return None

    async def fake_record_chat_answer(**_kwargs) -> None:
        raise AssertionError("cancelled answer should not be recorded")

    async def fake_greeting(_question: str) -> str:
        return ""

    async def run_case() -> None:
        task = asyncio.create_task(
            feishu._answer_feishu_message(
                question="question",
                sender_id="user-id",
                sender_name="User Name",
                chat_id="chat-id",
                message_id="incoming-message-id",
                event_id="event-id",
                chat_type="p2p",
                dedupe_key="message:incoming-message-id",
            )
        )
        await asyncio.wait_for(query_started.wait(), timeout=1)
        while not sent_cancel_ids:
            await asyncio.sleep(0)
        assert sent_cancel_ids[0] is not None
        cancelled = await feishu._cancel_feishu_answer(sent_cancel_ids[0])
        assert cancelled is True
        await asyncio.wait_for(task, timeout=1)

    monkeypatch.setattr(feishu, "_load_feedback_texts", lambda: ["step 1"])
    monkeypatch.setattr(feishu, "_generate_immediate_feedback_greeting", fake_greeting)
    monkeypatch.setattr(feishu, "_n8n_progress_polling_configured", lambda: False)
    monkeypatch.setattr(feishu, "_try_send_feishu_markdown_message", fake_send_message)
    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)
    monkeypatch.setattr(feishu, "ask_knowledge_base", fake_ask_knowledge_base)
    monkeypatch.setattr(feishu, "create_chat_record", fake_create_chat_record)
    monkeypatch.setattr(feishu, "record_chat_answer", fake_record_chat_answer)
    asyncio.run(run_case())

    assert any(canceled_text for _, _, canceled_text in updates)
    assert all(markdown_text != "final answer" for _, markdown_text, _ in updates)


def test_feishu_cancel_state_stays_available_for_stale_button_clicks(monkeypatch) -> None:
    updates: list[str | None] = []

    async def fake_update_message(
        message_id: str,
        markdown_text: str,
        *,
        canceled_text: str | None = None,
        **_kwargs,
    ) -> bool:
        updates.append(canceled_text)
        return True

    async def run_case() -> None:
        state = feishu._FeishuFeedbackState(
            message_id="feedback-message-id",
            question="question",
            prefix_text="prefix",
            status_text="status",
            update_lock=asyncio.Lock(),
            cancel_id="cancel-1",
        )
        handle = feishu._FeishuFeedbackHandle(
            state=state,
            stop_event=asyncio.Event(),
            task=asyncio.create_task(asyncio.sleep(30)),
            comfort_task=asyncio.create_task(asyncio.sleep(30)),
        )
        feishu._feishu_answer_cancel_states["cancel-1"] = feishu._FeishuAnswerCancelState(
            cancel_id="cancel-1",
            started_at=0.0,
            incoming_message_id="incoming-message-id",
            chat_id="chat-id",
            question="question",
            feedback_handle=handle,
            answer_task=None,
        )

        assert await feishu._cancel_feishu_answer("cancel-1") is True
        feishu._discard_feishu_answer_cancel_state("cancel-1")
        assert "cancel-1" in feishu._feishu_answer_cancel_states
        assert await feishu._cancel_feishu_answer("cancel-1") is True

        handle.task.cancel()
        handle.comfort_task.cancel()

    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)

    try:
        asyncio.run(run_case())
    finally:
        feishu._feishu_answer_cancel_states.pop("cancel-1", None)

    assert len(updates) >= 2
    assert all(canceled_text for canceled_text in updates)


def test_feishu_cancel_waits_for_feedback_tasks_before_cancel_card(monkeypatch) -> None:
    updates: list[str | None] = []
    feedback_task_cancelled = asyncio.Event()

    async def fake_update_message(
        message_id: str,
        markdown_text: str,
        *,
        canceled_text: str | None = None,
        **_kwargs,
    ) -> bool:
        updates.append(canceled_text)
        return True

    async def stale_feedback_task() -> None:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            feedback_task_cancelled.set()
            raise

    async def run_case() -> None:
        state = feishu._FeishuFeedbackState(
            message_id="feedback-message-id",
            question="question",
            prefix_text="prefix",
            status_text="status",
            update_lock=asyncio.Lock(),
            cancel_id="cancel-2",
        )
        handle = feishu._FeishuFeedbackHandle(
            state=state,
            stop_event=asyncio.Event(),
            task=asyncio.create_task(stale_feedback_task()),
            comfort_task=asyncio.create_task(asyncio.sleep(30)),
        )
        feishu._feishu_answer_cancel_states["cancel-2"] = feishu._FeishuAnswerCancelState(
            cancel_id="cancel-2",
            started_at=0.0,
            incoming_message_id="incoming-message-id",
            chat_id="chat-id",
            question="question",
            feedback_handle=handle,
            answer_task=None,
        )

        await asyncio.sleep(0)
        assert await feishu._cancel_feishu_answer("cancel-2") is True
        assert feedback_task_cancelled.is_set()
        assert handle.task.done()
        assert handle.comfort_task.done()

    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)

    try:
        asyncio.run(run_case())
    finally:
        feishu._feishu_answer_cancel_states.pop("cancel-2", None)

    assert len(updates) == 1
    assert updates[-1]


def test_feishu_card_action_response_contains_cancelled_card(monkeypatch) -> None:
    async def fake_update_message(*_args, **_kwargs) -> bool:
        return True

    async def run_case() -> dict:
        state = feishu._FeishuFeedbackState(
            message_id="feedback-message-id",
            question="question",
            prefix_text="prefix",
            status_text="status",
            update_lock=asyncio.Lock(),
            cancel_id="cancel-3",
        )
        handle = feishu._FeishuFeedbackHandle(
            state=state,
            stop_event=asyncio.Event(),
            task=asyncio.create_task(asyncio.sleep(30)),
            comfort_task=asyncio.create_task(asyncio.sleep(30)),
        )
        feishu._feishu_answer_cancel_states["cancel-3"] = feishu._FeishuAnswerCancelState(
            cancel_id="cancel-3",
            started_at=0.0,
            incoming_message_id="incoming-message-id",
            chat_id="chat-id",
            question="question",
            feedback_handle=handle,
            answer_task=None,
        )
        await asyncio.sleep(0)
        return await feishu._handle_feishu_card_action(
            {
                "event": {
                    "action": {
                        "value": {
                            "action": "stop_feishu_answer",
                            "cancel_id": "cancel-3",
                        }
                    }
                }
            }
        )

    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)

    try:
        result = asyncio.run(run_case())
    finally:
        feishu._feishu_answer_cancel_states.pop("cancel-3", None)

    assert result["cancelled"] is True
    assert result["card"]["type"] == "raw"
    elements = result["card"]["data"]["body"]["elements"]
    assert all(element.get("tag") != "button" for element in elements)
    assert elements[-2] == {"tag": "hr"}
