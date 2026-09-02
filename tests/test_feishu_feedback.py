from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from app.api.v1 import feishu
from app.schemas.query import QueryResponse
from app.services.privacy import encrypt_chat_text


def test_feishu_request_bypasses_environment_proxy_and_retries_connect_errors(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    sleeps: list[float] = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def request(self, method, url, **_kwargs):
            if len(calls) < 3:
                raise httpx.ConnectError(
                    "TLS proxy connection failed",
                    request=httpx.Request(method, url),
                )
            return httpx.Response(200, request=httpx.Request(method, url), json={"code": 0})

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(feishu.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(feishu.asyncio, "sleep", fake_sleep)

    response = asyncio.run(
        feishu._feishu_request(
            "POST",
            "https://open.feishu.cn/example",
            operation="test",
            json={"hello": "world"},
        )
    )

    assert response.status_code == 200
    assert len(calls) == 3
    assert all(call["trust_env"] is False for call in calls)
    assert sleeps == [0.5, 1.0]


def test_feishu_request_does_not_retry_read_timeout(monkeypatch) -> None:
    attempts = 0

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def request(self, method, url, **_kwargs):
            nonlocal attempts
            attempts += 1
            raise httpx.ReadTimeout("response lost", request=httpx.Request(method, url))

    monkeypatch.setattr(feishu.httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(
            feishu._feishu_request(
                "POST",
                "https://open.feishu.cn/example",
                operation="test",
            )
        )
    assert attempts == 1


def test_feishu_attachment_permission_error_is_actionable() -> None:
    response = httpx.Response(
        400,
        request=httpx.Request("GET", "https://open.feishu.cn/open-apis/im/v1/messages/x/resources/y"),
        json={
            "code": 99991672,
            "msg": "Access denied",
            "error": {"log_id": "log-test"},
        },
    )
    error = asyncio.run(feishu._feishu_attachment_response_error(response))

    assert error.code == 99991672
    assert "im:message:readonly" in str(error)
    assert "log-test" in str(error)
    assert "系统暂未获得读取飞书消息附件的权限" in error.user_message


def test_feishu_attachment_download_consumes_async_byte_stream(monkeypatch) -> None:
    captured: dict[str, object] = {}
    original_async_client = httpx.AsyncClient

    async def fake_token() -> str:
        return "tenant-token"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["type"] == "image"
        assert request.headers["authorization"] == "Bearer tenant-token"
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=b"image-bytes",
        )

    def client_factory(**_kwargs):
        return original_async_client(transport=httpx.MockTransport(handler))

    def fake_save_attachment(**kwargs):
        captured.update(kwargs)
        return {"attachment_id": "attachment-id", "filename": kwargs["filename"]}

    monkeypatch.setattr(feishu, "_get_tenant_access_token", fake_token)
    monkeypatch.setattr(feishu.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr("app.services.harness_attachments.save_attachment", fake_save_attachment)

    result = asyncio.run(feishu._download_and_store_feishu_attachment(
        message_id="message-id",
        attachment={
            "resource_key": "image-key",
            "resource_type": "image",
            "filename": "image-key.jpg",
        },
        user_id="user-id",
        internal_session_id="session-id",
    ))

    assert result["attachment_id"] == "attachment-id"
    assert captured["content"] == b"image-bytes"
    assert captured["content_type"] == "image/png"
    assert captured["filename"] == "image-key.png"


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


def test_image_message_is_queued_as_model_attachment(monkeypatch) -> None:
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
        "accepted": True,
        "event_id": "event-unsupported",
        "message_id": "message-unsupported",
    }
    assert len(queued_tasks) == 1
    func, kwargs = queued_tasks[0]
    assert func is feishu._answer_feishu_message
    assert kwargs["attachments"] == [{
        "resource_key": "img_v3_xyz",
        "resource_type": "image",
        "filename": "img_v3_xyz.jpg",
    }]


def test_bot_menu_daily_report_is_queued_for_requesting_user(monkeypatch) -> None:
    queued_tasks: list[tuple[object, dict[str, object]]] = []

    class FakeBackgroundTasks:
        def add_task(self, func, **kwargs) -> None:
            queued_tasks.append((func, kwargs))

    class FakeRequest:
        client = "test-client"

        async def json(self) -> dict:
            return {
                "header": {
                    "event_type": "application.bot.menu_v6",
                    "event_id": "menu-event-daily",
                },
                "event": {
                    "event_key": "report.daily",
                    "operator": {"operator_id": {"open_id": "ou_requester"}},
                },
            }

    monkeypatch.setattr(feishu, "_seen_message_keys", {})

    result = asyncio.run(feishu.handle_feishu_events(FakeRequest(), FakeBackgroundTasks()))

    assert result == {
        "ok": True,
        "accepted": True,
        "event_id": "menu-event-daily",
        "report_kind": "daily",
    }
    assert queued_tasks == [
        (
            feishu._send_feishu_menu_report,
            {
                "report_kind": "daily",
                "target_open_id": "ou_requester",
                "event_id": "menu-event-daily",
            },
        )
    ]


def test_bot_menu_weekly_report_is_queued_for_requesting_user(monkeypatch) -> None:
    queued_tasks: list[tuple[object, dict[str, object]]] = []

    class FakeBackgroundTasks:
        def add_task(self, func, **kwargs) -> None:
            queued_tasks.append((func, kwargs))

    class FakeRequest:
        client = "test-client"

        async def json(self) -> dict:
            return {
                "header": {
                    "event_type": "application.bot.menu_v6",
                    "event_id": "menu-event-weekly",
                },
                "event": {
                    "event_key": "report.weekly",
                    "operator": {"operator_id": {"open_id": "ou_requester"}},
                },
            }

    monkeypatch.setattr(feishu, "_seen_message_keys", {})

    result = asyncio.run(feishu.handle_feishu_events(FakeRequest(), FakeBackgroundTasks()))

    assert result["report_kind"] == "weekly"
    assert queued_tasks[0][1]["target_open_id"] == "ou_requester"


def test_extract_sender_identity_prefers_union_id_with_type() -> None:
    sender = {
        "sender_id": {
            "union_id": "user-union-id",
            "open_id": "user-open-id",
            "user_id": "user-id",
        }
    }

    assert feishu._extract_sender_id(sender) == "user-union-id"
    assert feishu._extract_sender_id_type(sender) == "union_id"


def test_answer_feishu_message_resolves_sender_name_from_contact_api(monkeypatch) -> None:
    captured_records: list[dict[str, object]] = []

    async def fake_resolve_sender_name(**kwargs):
        assert kwargs["sender_id"] == "user-open-id"
        assert kwargs["sender_id_type"] == "open_id"
        assert kwargs["sender_name"] is None
        return "张三"

    async def fake_create_chat_record(**kwargs) -> None:
        captured_records.append({"kind": "question", **kwargs})

    async def fake_record_chat_answer(**kwargs) -> None:
        captured_records.append({"kind": "answer", **kwargs})

    async def fake_ask_knowledge_base(_request, **_kwargs):
        return QueryResponse(question="question", answer="final answer")

    async def fake_resolve_feedback(*_args, **_kwargs):
        return None

    async def fake_stop_feedback(*_args, **_kwargs) -> None:
        return None

    async def fake_send_message(**_kwargs):
        return "sent-message-id"

    async def fake_register_feedback(_answer: str):
        return "feedback-id"

    monkeypatch.setattr(feishu, "_resolve_feishu_sender_name", fake_resolve_sender_name)
    monkeypatch.setattr(feishu, "_schedule_initial_feishu_feedback", lambda **_kwargs: None)
    monkeypatch.setattr(feishu, "_resolve_initial_feishu_feedback", fake_resolve_feedback)
    monkeypatch.setattr(feishu, "_stop_feishu_feedback_loop", fake_stop_feedback)
    monkeypatch.setattr(feishu, "_discard_feishu_answer_cancel_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(feishu, "_register_feishu_answer_feedback_state", fake_register_feedback)
    monkeypatch.setattr(feishu, "_try_send_feishu_markdown_message", fake_send_message)
    monkeypatch.setattr(feishu, "ask_knowledge_base", fake_ask_knowledge_base)
    monkeypatch.setattr(feishu, "create_chat_record", fake_create_chat_record)
    monkeypatch.setattr(feishu, "record_chat_answer", fake_record_chat_answer)

    asyncio.run(
        feishu._answer_feishu_message(
            question="question",
            sender_id="user-open-id",
            sender_id_type="open_id",
            sender_name=None,
            chat_id="chat-id",
            message_id="message-id",
            event_id="event-id",
            chat_type="p2p",
            dedupe_key="message:message-id",
        )
    )

    assert [record["kind"] for record in captured_records] == ["question", "answer"]
    assert captured_records[0]["user_name"] == "张三"
    assert captured_records[1]["user_name"] == "张三"


def test_extract_feishu_user_name_falls_back_to_i18n_name() -> None:
    assert feishu._extract_feishu_user_name(
        {"i18n_name": {"zh_cn": " 张三 ", "en_us": "Zhang San"}}
    ) == "张三"


def test_drive_file_edit_event_prints_raw_payload(capsys) -> None:
    class FakeBackgroundTasks:
        def add_task(self, func, **kwargs) -> None:  # pragma: no cover - should not run.
            raise AssertionError("drive file edit events should not queue message tasks yet")

    class FakeRequest:
        client = "test-client"

        async def json(self) -> dict:
            return {
                "header": {
                    "event_type": "drive.file.edit_v1",
                    "event_id": "event-drive-file-edit",
                },
                "event": {
                    "file_token": "doccn-example-token",
                    "file_type": "docx",
                    "operator_id": {"open_id": "ou-example"},
                },
            }

    result = asyncio.run(feishu.handle_feishu_events(FakeRequest(), FakeBackgroundTasks()))

    assert result == {
        "ok": True,
        "accepted": True,
        "event_type": "drive.file.edit_v1",
        "event_id": "event-drive-file-edit",
    }
    stdout = capsys.readouterr().out
    assert "drive.file.edit_v1 raw payload" in stdout
    assert "doccn-example-token" in stdout


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

    async def fake_ask_knowledge_base(_request, **_kwargs):
        await asyncio.sleep(0.16)
        return QueryResponse(question="question", answer="final answer")

    async def fake_create_chat_record(**_kwargs) -> None:
        return None

    async def fake_record_chat_answer(**_kwargs) -> None:
        return None

    async def fake_greeting(_question: str) -> str:
        return ""

    monkeypatch.setattr(feishu, "_feedback_interval_seconds", 0.05)
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

    assert sent_messages == [("chat-id", "incoming-message-id", "🤔 正在连接知识库助手...")]
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

    async def fake_ask_knowledge_base(_request, **_kwargs):
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

    async def fake_ask_knowledge_base(_request, **_kwargs):
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
    assert "send:🤔 正在连接知识库助手..." in events
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

    async def fake_ask_knowledge_base(_request, **_kwargs):
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

    assert "send:🤔 正在连接知识库助手..." in events
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

    async def fake_ask_knowledge_base(_request, **_kwargs):
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

    async def fake_ask_knowledge_base(_request, **_kwargs):
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
    assert "send:🤔 正在连接知识库助手..." in events


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


def test_upload_local_image_downloads_minio_image_before_feishu_upload(monkeypatch) -> None:
    calls = {}

    class FakeMinioClient:
        def fget_object(self, bucket, object_name, file_path):
            calls["minio"] = {
                "bucket": bucket,
                "object_name": object_name,
                "file_path": file_path,
            }
            with open(file_path, "wb") as image_file:
                image_file.write(b"png-bytes")

    class FakeResponse:
        status_code = 200
        text = '{"code":0,"data":{"image_key":"img_v3_minio"}}'

        def raise_for_status(self):
            pass

        def json(self):
            return {"code": 0, "data": {"image_key": "img_v3_minio"}}

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, url, headers, data, files):
            filename, image_file, mime_type = files["image"]
            calls["feishu"] = {
                "url": url,
                "headers": headers,
                "data": data,
                "filename": filename,
                "mime_type": mime_type,
                "bytes": image_file.read(),
                "temp_exists_during_upload": feishu.Path(image_file.name).is_file(),
                "temp_path": image_file.name,
            }
            return FakeResponse()

    monkeypatch.setattr(feishu, "get_minio_client", lambda: FakeMinioClient())
    monkeypatch.setattr(feishu.httpx, "AsyncClient", FakeAsyncClient)

    image_key = asyncio.run(
        feishu._upload_local_image(
            "minio://knowledge-processed-docs/%E8%BF%88%E6%8B%93%E6%80%9D%E5%AD%A6%E9%99%A2/"
            "%E6%88%90%E9%95%BF%E6%89%8B%E5%86%8C%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.docx/"
            "img/image_0002.png",
            "tenant-token",
        )
    )

    assert image_key == "img_v3_minio"
    assert calls["minio"]["bucket"] == "knowledge-processed-docs"
    assert calls["minio"]["object_name"] == "迈拓思学院/成长手册使用说明.docx/img/image_0002.png"
    assert calls["feishu"]["bytes"] == b"png-bytes"
    assert calls["feishu"]["filename"].endswith(".png")
    assert calls["feishu"]["mime_type"] == "image/png"
    assert calls["feishu"]["temp_exists_during_upload"] is True
    assert not feishu.Path(calls["feishu"]["temp_path"]).exists()


def test_download_minio_image_to_temp_decodes_standard_asset_uri(monkeypatch) -> None:
    calls = {}

    class FakeMinioClient:
        def fget_object(self, bucket, object_name, file_path):
            calls["bucket"] = bucket
            calls["object_name"] = object_name
            with open(file_path, "wb") as image_file:
                image_file.write(b"standard-image")

    monkeypatch.setattr(feishu, "get_minio_client", lambda: FakeMinioClient())

    temp_path = feishu._download_minio_image_to_temp(
        "minio://knowledge-standard-assets/%E4%BA%A7%E5%93%81%E6%A0%87%E5%87%86/"
        "ASME-Sec-II-A-Vol1-2023%28%E5%88%87%E5%88%86%E7%89%88%29/"
        "SA-213_SA-213M%20-%20SPECIFICATION/table%20-%20TABLE%201%20Chemical%20Composition"
        "%20Limits%2C%20%25%20A%20%2C%20for%20Low%20Alloy%20Steel.png"
    )

    try:
        assert temp_path is not None
        assert temp_path.read_bytes() == b"standard-image"
        assert calls["bucket"] == "knowledge-standard-assets"
        assert calls["object_name"] == (
            "产品标准/ASME-Sec-II-A-Vol1-2023(切分版)/"
            "SA-213_SA-213M - SPECIFICATION/table - TABLE 1 Chemical Composition"
            " Limits, % A , for Low Alloy Steel.png"
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def test_feishu_card_flattens_markdown_tables_for_json2_renderer() -> None:
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
    rendered = card["body"]["elements"][0]["content"]
    assert "**标题**" in rendered
    assert "▌ 引用内容" in rendered
    assert "1. 列一: A；列二: B" in rendered
    assert "| --- |" not in rendered


def test_feishu_card_flattens_many_tables_before_delivery() -> None:
    markdown = "\n\n".join(
        f"表{i}\n名称 | 数值\n--- | ---\n项目{i} | {i}"
        for i in range(20)
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))
    card = json.loads(content)
    rendered = card["body"]["elements"][0]["content"]

    assert "--- | ---" not in rendered
    assert rendered.count("名称: 项目") == 20


def test_feishu_card_keeps_terminal_bold_span_renderable() -> None:
    markdown = (
        "**MT Pipeline（管道）**、**Oilfield Services（油服/盘管）**、"
        "**MT Wire（线材）**、**MT Plate（板材）**"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))
    card = json.loads(content)
    rendered = card["body"]["elements"][0]["content"]

    assert rendered[:-1] == markdown
    assert rendered.endswith("**\u200b")


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


def test_feishu_card_renders_display_latex_dollar_delimiters() -> None:
    markdown = r"$$$$\frac{26.5}{7.10\times 0.87} = 4.663$$$$"

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert card["body"]["elements"][0]["content"] == "(26.5) / (7.10 x 0.87) = 4.663"


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


def test_feishu_card_renders_fraction_with_nested_text_groups() -> None:
    markdown = (
        r"[\text{FOB USD/KG}=\frac{\text{出厂价}+\text{运营成本}}"
        r"{\text{汇率}\times(1-\text{利润率})}]"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert (
        card["body"]["elements"][0]["content"]
        == "FOB USD/KG = (出厂价 + 运营成本) / (汇率 x (1 - 利润率))"
    )


def test_feishu_card_does_not_expose_unmatched_frac_command() -> None:
    markdown = r"[\frac出厂价+\frac国内运杂费/1000汇率]"

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert "frac" not in card["body"]["elements"][0]["content"]


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


def test_harness_progress_merges_tool_wave_and_uses_neutral_copy(monkeypatch) -> None:
    updates: list[str] = []

    async def fake_update(_message_id: str, markdown_text: str, **_kwargs) -> bool:
        updates.append(markdown_text)
        return True

    async def run_case() -> feishu._FeishuFeedbackState:
        state = feishu._FeishuFeedbackState(
            message_id="feedback-message-id",
            question="question",
            prefix_text="",
            status_text="",
            update_lock=asyncio.Lock(),
            harness_mode=True,
            progress_lines=["<font color='grey'>⏳ 正在连接知识库助手…</font>"],
        )
        await feishu._append_harness_progress(
            state, SimpleNamespace(kind="status", tool_name="", text="正在准备检索")
        )
        await feishu._append_harness_progress(
            state, SimpleNamespace(kind="text", tool_name="", text="我来帮您查询壁厚单位及其转换规则。")
        )
        await feishu._append_harness_progress(
            state, SimpleNamespace(kind="tool_start", tool_name="kb_hybrid_search", arguments={"query": "**secret**"})
        )
        await feishu._append_harness_progress(
            state, SimpleNamespace(kind="tool_end", tool_name="kb_hybrid_search", result={})
        )
        await feishu._append_harness_progress(
            state, SimpleNamespace(kind="tool_start", tool_name="kb_graph_search", arguments={})
        )
        await feishu._append_harness_progress(
            state, SimpleNamespace(kind="tool_end", tool_name="kb_graph_search", result={})
        )
        await feishu._append_harness_progress(
            state, SimpleNamespace(kind="status", tool_name="", text="正在组织答案")
        )
        return state

    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update)
    state = asyncio.run(run_case())

    assert state.progress_lines == [
        "💬 我来帮您查询壁厚单位及其转换规则。",
        "<font color='grey'>✓ 已完成：混合检索 1 次、知识图谱检索 1 次</font>",
        "✦ 正在组织答案",
    ]
    assert all("color='green'" not in update and "color='blue'" not in update for update in updates)
    assert all("secret" not in update for update in updates)
    assert all("•" not in update for update in updates)
    assert all("正在准备检索" not in update for update in updates[-4:])
    assert feishu._clean_harness_feedback_text(r"处理中 \*\*重点\*\*") == "处理中 重点"


def test_harness_progress_uses_bounded_table_safe_rolling_window(monkeypatch) -> None:
    updates: list[str] = []

    async def fake_update(_message_id: str, markdown_text: str, **_kwargs) -> bool:
        updates.append(markdown_text)
        return True

    async def run_case() -> feishu._FeishuFeedbackState:
        state = feishu._FeishuFeedbackState(
            message_id="feedback-message-id",
            question="question",
            prefix_text="",
            status_text="",
            update_lock=asyncio.Lock(),
            harness_mode=True,
        )
        for index in range(15):
            await feishu._append_harness_progress(
                state,
                SimpleNamespace(
                    kind="text",
                    tool_name="",
                    text=f"步骤{index} | 表头 | 内容 " + ("很长" * 80),
                ),
            )
        return state

    monkeypatch.setattr(feishu, "_try_update_feishu_markdown_message", fake_update)
    state = asyncio.run(run_case())

    assert len(state.progress_lines) <= 6
    assert sum(len(line) + 1 for line in state.progress_lines) <= 1_200
    assert all("|" not in line for line in state.progress_lines)
    assert "步骤14" in state.progress_lines[-1]


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


def test_feishu_card_hides_unmapped_reference_tags() -> None:
    markdown = "Answer from KB<reference>data/raw/a.docx</Reference>"

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    elements = card["body"]["elements"]
    assert elements[0]["content"] == "Answer from KB"
    assert len(elements) == 1


def test_feishu_card_hides_repeated_unmapped_reference_tags() -> None:
    markdown = (
        "First<reference>data/raw/a.docx</reference>\n"
        "Second<Reference>data\\raw\\a.docx</Reference>"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    elements = card["body"]["elements"]
    assert elements[0]["content"] == "First\nSecond"
    assert len(elements) == 1


def test_feishu_card_hides_unmapped_reference_tags_split_across_lines() -> None:
    markdown = (
        "结论内容。<reference>data/raw/a.docx</reference>\n"
        "<Reference>data/raw/b.docx</Reference>\n"
        "<reference>data/raw/c.docx</reference>"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    main_content = card["body"]["elements"][0]["content"]
    assert main_content == "结论内容。"


def test_feishu_card_hides_internal_download_references() -> None:
    url_a = feishu._build_document_download_url("data/raw/a.docx")
    url_b = feishu._build_document_download_url("data/raw/b.docx")
    markdown = f"结论内容。[[1]]({url_a})\n[[2]]({url_b})"

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert card["body"]["elements"][0]["content"] == "结论内容。"


def test_feishu_card_redacts_internal_paths_inside_code_blocks() -> None:
    markdown = "```\n<reference>data/raw/a.docx</reference>\n```"

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert card["body"]["elements"][0]["content"] == "```\n<reference>[内部信息已隐藏]</reference>\n```"
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


def test_feishu_card_keeps_model_source_explanation_but_hides_unmapped_paths() -> None:
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
    assert elements[0]["content"] == "第一段。来源甲\n第二段。重复来源\n\n────────\n知识来源：\n来源甲"
    assert len(elements) == 1


def test_feishu_card_keeps_descriptions_for_unmapped_reference_links() -> None:
    markdown = (
        "结论内容。"
        "[片段7,来源甲](data/raw/a.docx)"
        "[片段2,重复来源](data/raw/a.docx)"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    elements = card["body"]["elements"]
    assert elements[0]["content"] == "结论内容。来源甲重复来源"
    assert len(elements) == 1


def test_feishu_card_removes_adjacent_internal_download_references() -> None:
    existing_url = feishu._build_document_download_url("data/raw/a.docx")
    markdown = f"结论内容。[[1]]({existing_url})[[1]]({existing_url})"

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    card = json.loads(content)
    assert card["body"]["elements"][0]["content"] == "结论内容。"


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


def test_feishu_card_redacts_internal_details_and_keeps_public_links() -> None:
    markdown = (
        "内部位置 minio://knowledge-raw-docs/private/manual.docx，"
        "工具 kb_hybrid_search，"
        "服务 http://127.0.0.1:8000/internal，"
        "公开来源：[官网](https://example.com/docs)。"
    )

    content = asyncio.run(feishu._build_feishu_card_content(markdown, "tenant-token"))

    rendered = json.loads(content)["body"]["elements"][0]["content"]
    assert "minio://" not in rendered
    assert "knowledge-raw-docs" not in rendered
    assert "kb_hybrid_search" not in rendered
    assert "127.0.0.1" not in rendered
    assert "[官网](https://example.com/docs)" in rendered


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


def test_format_markdown_link_url_hides_unmapped_internal_path(monkeypatch) -> None:
    monkeypatch.setattr(feishu, "_local_to_lark_mapping_cache", {})
    raw_path = "data/raw/folder/missing.docx"

    assert feishu._format_markdown_link_url(raw_path) == ""


def test_format_markdown_link_url_keeps_existing_lark_url() -> None:
    raw_url = "https://example.feishu.cn/wiki/demo"

    assert feishu._format_markdown_link_url(raw_url) == raw_url


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

    async def fake_ask_knowledge_base(_request, **_kwargs):
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
