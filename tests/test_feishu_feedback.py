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
    monkeypatch.setattr(feishu, "_n8n_progress_polling_configured", lambda: False)
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

    assert events == [
        "send:step 1",
        "query",
        "record-answer",
        "update:final answer",
        "schedule-evaluation",
    ]


def test_feishu_card_images_render_as_json2_image_elements(monkeypatch) -> None:
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
    assert card["schema"] == "2.0"
    elements = card["body"]["elements"]
    assert elements[0] == {
        "tag": "markdown",
        "content": "before",
        "text_align": "left",
        "text_size": "normal_v2",
    }
    assert elements[1] == {
        "tag": "img",
        "img_key": "img_v3_key",
        "alt": {
            "tag": "plain_text",
            "content": "screenshot.png",
        },
    }
    assert elements[2] == {
        "tag": "markdown",
        "content": "after",
        "text_align": "left",
        "text_size": "normal_v2",
    }


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
    expected_url = feishu._build_document_download_url(
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
    expected_url = feishu._build_document_download_url(
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
    expected_url = feishu._build_document_download_url(
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
