from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from uuid import UUID

from app.core.config import settings
from app.db.postgres import (
    assign_latest_chat_message_topic,
    create_conversation_topic,
    ensure_chat_messages_table,
    ensure_conversation_topics_table,
    get_conversation_topic,
    list_chat_messages_by_topic,
    list_recent_conversation_topics,
    update_conversation_topic,
)
from app.services.chat_records import normalize_chat_id, normalize_optional_text
from app.services.privacy import decrypt_chat_text, encrypt_chat_text


def _normalize_topic_text(value: str | None, fallback: str) -> str:
    cleaned = str(value).strip() if value is not None else ""
    return cleaned or fallback


def _looks_like_corrupted_text(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if "\ufffd" in text:
        return True
    non_space = "".join(ch for ch in text if not ch.isspace())
    return bool(non_space) and set(non_space) <= {"?"}


def _require_valid_topic_text(field_name: str, value: str | None) -> None:
    if value is not None and _looks_like_corrupted_text(value):
        raise ValueError(
            f"conversation topic {field_name} looks corrupted; send UTF-8 JSON text"
        )


async def create_conversation_topic_record(
    *,
    user_id: str | None,
    session_id: str | None,
    topic: str,
    summary: str = "",
    topic_id: UUID | str | None = None,
    started_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an encrypted virtual topic for one Feishu user/session pair."""

    return await asyncio.to_thread(
        _create_conversation_topic_record_sync,
        user_id=normalize_chat_id(user_id, fallback="unknown-user"),
        session_id=normalize_chat_id(session_id, fallback="unknown-session"),
        topic=_normalize_topic_text(topic, fallback="未命名话题"),
        summary=normalize_optional_text(summary) or "",
        topic_id=topic_id,
        started_at=started_at,
        metadata=metadata or {},
    )


async def update_conversation_topic_record(
    *,
    topic_id: UUID | str,
    topic: str | None = None,
    summary: str | None = None,
    message_increment: int = 0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update an existing virtual topic, encrypting topic and summary changes."""

    return await asyncio.to_thread(
        _update_conversation_topic_record_sync,
        topic_id=topic_id,
        topic=normalize_optional_text(topic),
        summary=normalize_optional_text(summary),
        message_increment=message_increment,
        metadata=metadata,
    )


async def update_conversation_topic_summary(
    *,
    topic_id: UUID | str,
    user_id: str | None,
    session_id: str | None,
    summary: str,
    topic: str | None = None,
    conversation_id: str | None = None,
    message_increment: int = 0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update a topic summary and optionally attach the latest chat row to it."""

    normalized_session_id = normalize_chat_id(session_id, fallback="unknown-session")
    return await asyncio.to_thread(
        _update_conversation_topic_summary_sync,
        topic_id=topic_id,
        user_id=normalize_chat_id(user_id, fallback="unknown-user"),
        session_id=normalized_session_id,
        conversation_id=(
            normalize_chat_id(conversation_id, fallback=normalized_session_id)
            if conversation_id is not None
            else None
        ),
        topic=normalize_optional_text(topic),
        summary=_normalize_topic_text(summary, fallback=""),
        message_increment=message_increment,
        metadata=metadata,
    )


async def list_recent_conversation_topic_records(
    *,
    user_id: str | None,
    session_id: str | None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """List recent active topics with decrypted topic and summary fields."""

    return await asyncio.to_thread(
        _list_recent_conversation_topic_records_sync,
        user_id=normalize_chat_id(user_id, fallback="unknown-user"),
        session_id=normalize_chat_id(session_id, fallback="unknown-session"),
        limit=limit or settings.conversation_topic_recent_limit,
    )


async def get_conversation_topic_context(
    *,
    topic_id: UUID | str,
    user_id: str | None,
    session_id: str | None,
    message_limit: int = 10,
) -> dict[str, Any]:
    """Return one topic's summary plus recent decrypted chat turns."""

    return await asyncio.to_thread(
        _get_conversation_topic_context_sync,
        topic_id=topic_id,
        user_id=normalize_chat_id(user_id, fallback="unknown-user"),
        session_id=normalize_chat_id(session_id, fallback="unknown-session"),
        message_limit=message_limit,
    )


def _create_conversation_topic_record_sync(
    *,
    user_id: str,
    session_id: str,
    topic: str,
    summary: str,
    topic_id: UUID | str | None,
    started_at: datetime | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    ensure_conversation_topics_table()
    _require_valid_topic_text("topic", topic)
    _require_valid_topic_text("summary", summary)
    row = create_conversation_topic(
        user_id=user_id,
        session_id=session_id,
        topic=encrypt_chat_text(topic),
        summary=encrypt_chat_text(summary),
        topic_id=topic_id,
        started_at=started_at,
        metadata=metadata,
    )
    return _decrypt_topic_row(row)


def _update_conversation_topic_record_sync(
    *,
    topic_id: UUID | str,
    topic: str | None,
    summary: str | None,
    message_increment: int,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    ensure_conversation_topics_table()
    _require_valid_topic_text("topic", topic)
    _require_valid_topic_text("summary", summary)
    row = update_conversation_topic(
        topic_id=topic_id,
        topic=encrypt_chat_text(topic) if topic is not None else None,
        summary=encrypt_chat_text(summary) if summary is not None else None,
        message_increment=message_increment,
        metadata=metadata,
    )
    return _decrypt_topic_row(row)


def _update_conversation_topic_summary_sync(
    *,
    topic_id: UUID | str,
    user_id: str,
    session_id: str,
    conversation_id: str | None,
    topic: str | None,
    summary: str,
    message_increment: int,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    ensure_conversation_topics_table()
    ensure_chat_messages_table()
    _require_valid_topic_text("topic", topic)
    _require_valid_topic_text("summary", summary)
    existing_topic = get_conversation_topic(
        topic_id=topic_id,
        user_id=user_id,
        session_id=session_id,
    )
    if not existing_topic:
        return {}

    row = update_conversation_topic(
        topic_id=topic_id,
        topic=encrypt_chat_text(topic) if topic is not None else None,
        summary=encrypt_chat_text(summary),
        message_increment=message_increment,
        metadata=metadata,
    )
    topic_row = _decrypt_topic_row(row)
    if not topic_row:
        return {}

    bound_message: dict[str, Any] = {}
    if conversation_id is not None:
        bound_message = assign_latest_chat_message_topic(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            topic_id=topic_id,
        )

    return {
        "topic": topic_row,
        "bound_message": _decrypt_message_row(bound_message),
    }


def _list_recent_conversation_topic_records_sync(
    *,
    user_id: str,
    session_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    ensure_conversation_topics_table()
    rows = list_recent_conversation_topics(
        user_id=user_id,
        session_id=session_id,
        limit=max(limit, min(limit * 3, 50)),
    )
    topics = [_decrypt_topic_row(row) for row in rows]
    return [
        row
        for row in topics
        if not _topic_row_has_corrupted_text(row)
    ][:limit]


def _get_conversation_topic_context_sync(
    *,
    topic_id: UUID | str,
    user_id: str,
    session_id: str,
    message_limit: int,
) -> dict[str, Any]:
    ensure_conversation_topics_table()
    ensure_chat_messages_table()
    topic = get_conversation_topic(
        topic_id=topic_id,
        user_id=user_id,
        session_id=session_id,
    )
    if not topic:
        return {}
    messages = list_chat_messages_by_topic(
        topic_id=topic_id,
        user_id=user_id,
        session_id=session_id,
        limit=message_limit,
    )
    return {
        "topic": _decrypt_topic_row(topic),
        "messages": [_decrypt_message_row(row) for row in messages],
    }


def _topic_row_has_corrupted_text(row: dict[str, Any]) -> bool:
    return _looks_like_corrupted_text(str(row.get("topic") or "")) or _looks_like_corrupted_text(
        str(row.get("summary") or "")
    )


def _decrypt_topic_row(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    decrypted = dict(row)
    decrypted["topic"] = decrypt_chat_text(str(decrypted.get("topic") or ""))
    decrypted["summary"] = decrypt_chat_text(str(decrypted.get("summary") or ""))
    return decrypted


def _decrypt_message_row(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    decrypted = dict(row)
    decrypted["question"] = decrypt_chat_text(str(decrypted.get("question") or ""))
    decrypted["answer"] = decrypt_chat_text(str(decrypted.get("answer") or ""))
    return decrypted
