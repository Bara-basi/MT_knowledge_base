from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID

from app.api.v1 import external
from app.schemas.external import ExternalQueryRequest
from app.schemas.query import QueryResponse
from app.services import external_answer_formatting as formatting
from app.services.chat import topics
from app.services.privacy import encrypt_chat_text
from app.services.external_chat_records import canonical_external_ids


def test_external_ids_are_service_scoped_and_do_not_store_raw_identity() -> None:
    first = canonical_external_ids(
        service_id="crm",
        user_id="employee@example.com",
        session_id="ticket-42",
    )
    same = canonical_external_ids(
        service_id="crm",
        user_id="employee@example.com",
        session_id="ticket-42",
    )
    other_service = canonical_external_ids(
        service_id="erp",
        user_id="employee@example.com",
        session_id="ticket-42",
    )

    assert first == same
    assert first != other_service
    assert first["user_id"].startswith("external:v1:u:")
    assert first["session_id"].startswith("external:v1:s:")
    assert "employee@example.com" not in first["user_id"]
    assert "ticket-42" not in first["session_id"]
    assert len(first["user_id"]) <= 128


def test_external_markdown_rewrites_and_deduplicates_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        formatting,
        "settings",
        SimpleNamespace(
            public_base_url="https://kb.example.com/prod",
            api_route_prefix="/prod",
        ),
    )
    answer = (
        "结论。<reference>data/raw/制度.docx</reference>\n\n"
        "补充。<reference>data/raw/制度.docx</reference>\n\n"
        "<img>data/raw/流程.png</img>"
    )

    rendered = formatting.format_external_markdown_answer(
        answer,
        use_lark_document=False,
    )

    assert "结论。[1]" in rendered
    assert "补充。[1]" in rendered
    assert rendered.count("1. [制度.docx]") == 1
    assert "\n\n---\n### 知识来源\n" in rendered
    assert "![流程.png](https://kb.example.com/prod/api/v1/documents/download?" in rendered
    assert "<reference>" not in rendered


def test_external_markdown_prefers_lark_mapping(monkeypatch) -> None:
    monkeypatch.setattr(
        formatting,
        "_local_to_lark_mapping_cache",
        {"制度.docx": "https://example.feishu.cn/docx/abc"},
    )

    rendered = formatting.format_external_markdown_answer(
        "依据。<reference>data/raw/制度.docx</reference>",
        use_lark_document=True,
    )

    assert "[制度.docx](https://example.feishu.cn/docx/abc)" in rendered


def test_external_query_uses_isolated_n8n_identity_and_storage(monkeypatch) -> None:
    captured: dict = {}
    message_id = UUID("00000000-0000-0000-0000-000000000010")
    topic_id = "00000000-0000-0000-0000-000000000020"

    async def fake_create(**kwargs):
        captured["create"] = kwargs
        return {"message_id": message_id}

    async def fake_ask(request):
        captured["n8n"] = request
        return QueryResponse(
            question=request.question,
            answer="答案。<reference>data/raw/制度.docx</reference>",
            topic_id=topic_id,
        )

    async def fake_record(**kwargs):
        captured["record"] = kwargs
        return kwargs

    monkeypatch.setattr(external, "create_external_chat_record", fake_create)
    monkeypatch.setattr(external, "ask_knowledge_base", fake_ask)
    monkeypatch.setattr(external, "record_external_chat_answer", fake_record)

    response = asyncio.run(
        external.query_external_knowledge_base(
            ExternalQueryRequest(
                question="制度是什么？",
                user_id="user-1",
                service_id="crm",
                session_id="session-1",
                use_lark_document=False,
            )
        )
    )

    assert response.user_id == "user-1"
    assert response.session_id == "session-1"
    assert response.service_id == "crm"
    assert response.answer_format == "markdown"
    assert "---\n### 知识来源" in response.answer
    assert captured["create"]["user_id"].startswith("external:v1:u:")
    assert captured["create"]["session_id"].startswith("external:v1:s:")
    assert captured["n8n"].source == "external"
    assert captured["n8n"].metadata == {
        "source": "external",
        "service_id": "crm",
        "format_type": "markdown",
    }
    assert captured["record"]["message_id"] == message_id
    assert captured["record"]["topic_id"] == topic_id


def test_external_topic_context_reads_only_external_message_table(monkeypatch) -> None:
    user_id = canonical_external_ids(
        service_id="crm",
        user_id="user-1",
        session_id="session-1",
    )["user_id"]
    session_id = canonical_external_ids(
        service_id="crm",
        user_id="user-1",
        session_id="session-1",
    )["session_id"]
    captured: dict = {}

    monkeypatch.setattr(topics, "ensure_conversation_topics_table", lambda: None)
    monkeypatch.setattr(
        topics,
        "ensure_external_chat_messages_table",
        lambda table_name: captured.setdefault("ensured_table", table_name),
    )
    monkeypatch.setattr(
        topics,
        "ensure_chat_messages_table",
        lambda: (_ for _ in ()).throw(AssertionError("Feishu table must not be used")),
    )
    monkeypatch.setattr(
        topics,
        "get_conversation_topic",
        lambda **kwargs: {
            "topic_id": kwargs["topic_id"],
            "user_id": kwargs["user_id"],
            "session_id": kwargs["session_id"],
            "topic": encrypt_chat_text("外部主题"),
            "summary": encrypt_chat_text("外部总结"),
        },
    )

    def fake_list_messages(**kwargs):
        captured["list"] = kwargs
        return [
            {
                "user_id": user_id,
                "user_name": None,
                "session_id": session_id,
                "conversation_id": session_id,
                "topic_id": kwargs["topic_id"],
                "create_time": None,
                "question": encrypt_chat_text("外部问题"),
                "answer": encrypt_chat_text("外部回答"),
            }
        ]

    monkeypatch.setattr(topics, "list_chat_messages_by_topic", fake_list_messages)

    context = topics._get_conversation_topic_context_sync(
        topic_id="00000000-0000-0000-0000-000000000020",
        user_id=user_id,
        session_id=session_id,
        message_limit=6,
    )

    assert captured["ensured_table"] == "chat_messages_external"
    assert captured["list"]["table_name"] == "chat_messages_external"
    assert context["messages"][0]["question"] == "外部问题"
