from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.api.v1 import feishu
from app.schemas.query import QueryResponse


@pytest.fixture(autouse=True)
def disable_real_feedback_stream(monkeypatch):
    async def fake_stream(_question: str):
        raise feishu.LLMAPIError("stream disabled in tests")
        yield ""

    monkeypatch.setattr(feishu, "_stream_immediate_feedback_greeting", fake_stream)


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

    reply_event = "update:final answer" if "update:final answer" in events else "send:final answer"
    assert events.index(reply_event) < events.index("schedule-evaluation")
    assert events[-1] == "schedule-evaluation"


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


def test_streaming_initial_feedback_updates_same_card(monkeypatch) -> None:
    events: list[str] = []

    async def fake_stream(_question: str):
        yield "收到，"
        await asyncio.sleep(0)
        yield "我先查一下资料。"

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

    monkeypatch.setattr(feishu, "_stream_immediate_feedback_greeting", fake_stream)
    monkeypatch.setattr(feishu, "_stream_update_min_interval_seconds", 0)
    monkeypatch.setattr(feishu, "_load_feedback_texts", lambda: ["🤔 正在理解问题..."])
    monkeypatch.setattr(feishu, "_try_send_feishu_markdown_message", fake_send_message)
    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)

    result = asyncio.run(
        feishu._send_streaming_initial_feedback_card(
            question="question",
            chat_id="chat-id",
            message_id="incoming-message-id",
        )
    )

    assert result == ("feedback-message-id", "收到，我先查一下资料。", "🤔 正在理解问题...")
    assert events[0] == "send:收到，\n\n🤔 正在理解问题..."
    assert events[-1] == "update:收到，我先查一下资料。\n\n🤔 正在理解问题..."


def test_pseudo_stream_final_answer_updates_in_chunks(monkeypatch) -> None:
    updates: list[str] = []

    async def fake_update_message(message_id: str, markdown_text: str, **_kwargs) -> bool:
        assert message_id == "answer-message-id"
        updates.append(markdown_text)
        return True

    answer = "第一段内容。" * 40
    monkeypatch.setattr(feishu, "_pseudo_stream_update_delay_seconds", 0)
    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)

    sent = asyncio.run(
        feishu._pseudo_stream_feishu_final_answer(
            message_id="answer-message-id",
            answer=answer,
        )
    )

    assert sent is True
    assert len(updates) > 1
    assert updates[-1] == answer


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
    ):
        assert question == "question"
        assert minute == 1
        assert status_text == "🤔 正在理解问题..."
        yield "这个问题我还在对照资料，"
        await asyncio.sleep(0)
        yield "马上整理给你。"

    monkeypatch.setattr(feishu, "_feedback_interval_seconds", 10)
    monkeypatch.setattr(feishu, "_comfort_feedback_interval_seconds", 0.05)
    monkeypatch.setattr(feishu, "_n8n_progress_polling_configured", lambda: False)
    monkeypatch.setattr(feishu, "_load_feedback_texts", lambda: ["🤔 正在理解问题..."])
    monkeypatch.setattr(feishu, "_generate_immediate_feedback_greeting", fake_greeting)
    monkeypatch.setattr(feishu, "_stream_comfort_feedback_text", fake_comfort_feedback_text)
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
    monkeypatch.setattr(
        feishu,
        "_schedule_feishu_answer_evaluation",
        lambda **_kwargs: events.append("schedule-evaluation"),
    )

    asyncio.run(
        feishu._answer_feishu_message(
            question="你好",
            sender_id="user-id",
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

    def fail_schedule_evaluation(**_kwargs) -> None:
        raise AssertionError("transient progress text should not schedule evaluation")

    monkeypatch.setattr(feishu, "_load_feedback_texts", lambda: ["正在整理资料..."])
    monkeypatch.setattr(feishu, "_generate_immediate_feedback_greeting", fake_greeting)
    monkeypatch.setattr(feishu, "_n8n_progress_polling_configured", lambda: False)
    monkeypatch.setattr(feishu, "_try_send_feishu_markdown_message", fake_send_message)
    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update_message)
    monkeypatch.setattr(feishu, "ask_knowledge_base", fake_ask_knowledge_base)
    monkeypatch.setattr(feishu, "create_chat_record", fake_create_chat_record)
    monkeypatch.setattr(feishu, "record_chat_answer", fail_record_chat_answer)
    monkeypatch.setattr(feishu, "_schedule_feishu_answer_evaluation", fail_schedule_evaluation)

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


def test_n8n_progress_stage_advances_after_rewrite_finishes() -> None:
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


def test_n8n_progress_stage_returns_latest_mapped_stage() -> None:
    execution = {
        "workflowData": {
            "nodes": [
                {
                    "id": "54aad033-e2d6-4b2a-aa73-027f9fc839ce",
                    "name": "renamed rewrite node",
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
                    "renamed rewrite node": [
                        {"startTime": "2026-06-15T01:00:00.000Z"}
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
