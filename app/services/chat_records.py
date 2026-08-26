from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.db.postgres import (
    create_chat_message,
    ensure_chat_messages_table,
    insert_chat_message,
    update_chat_answer,
)
from app.services.privacy import encrypt_chat_text


logger = logging.getLogger(__name__)


def normalize_chat_id(value: str | None, fallback: str) -> str:
    cleaned = str(value).strip() if value is not None else ""
    return cleaned or fallback


async def create_chat_record(
    *,
    user_id: str | None,
    session_id: str | None,
    conversation_id: str | None,
    question: str,
    user_name: str | None = None,
    source_message_id: str | None = None,
) -> dict[str, Any]:
    """Persist the initial user question before the answer is available."""

    ids = normalize_record_ids(
        user_id=user_id,
        session_id=session_id,
        conversation_id=conversation_id,
    )
    return await asyncio.to_thread(
        _create_chat_record_sync,
        user_id=ids["user_id"],
        user_name=normalize_optional_text(user_name),
        source_message_id=normalize_optional_text(source_message_id),
        session_id=ids["session_id"],
        conversation_id=ids["conversation_id"],
        question=question,
    )


async def record_chat_answer(
    *,
    message_id: str | None = None,
    user_id: str | None,
    session_id: str | None,
    conversation_id: str | None,
    question: str,
    answer: str,
    user_name: str | None = None,
    topic_id: str | None = None,
) -> dict[str, Any]:
    """Persist the workflow answer to the existing user question row."""

    ids = normalize_record_ids(
        user_id=user_id,
        session_id=session_id,
        conversation_id=conversation_id,
    )
    return await asyncio.to_thread(
        _record_chat_answer_sync,
        message_id=message_id,
        user_id=ids["user_id"],
        user_name=normalize_optional_text(user_name),
        session_id=ids["session_id"],
        conversation_id=ids["conversation_id"],
        question=question,
        answer=answer,
        topic_id=None,
    )


def normalize_record_ids(
    *,
    user_id: str | None,
    session_id: str | None,
    conversation_id: str | None,
) -> dict[str, str]:
    session = normalize_chat_id(session_id, fallback="unknown-session")
    return {
        "user_id": normalize_chat_id(user_id, fallback="unknown-user"),
        "session_id": session,
        "conversation_id": normalize_chat_id(conversation_id, fallback=session),
    }


def normalize_optional_text(value: str | None) -> str | None:
    cleaned = str(value).strip() if value is not None else ""
    return cleaned or None


def _create_chat_record_sync(
    *,
    user_id: str,
    user_name: str | None,
    source_message_id: str | None,
    session_id: str,
    conversation_id: str,
    question: str,
) -> dict[str, Any]:
    ensure_chat_messages_table()
    return create_chat_message(
        user_id=user_id,
        user_name=user_name,
        source_message_id=source_message_id,
        session_id=session_id,
        conversation_id=conversation_id,
        question=encrypt_chat_text(question),
    )


def _record_chat_answer_sync(
    *,
    message_id: str | None,
    user_id: str,
    user_name: str | None,
    session_id: str,
    conversation_id: str,
    question: str,
    answer: str,
    topic_id: str | None = None,
) -> dict[str, Any]:
    ensure_chat_messages_table()
    row = update_chat_answer(
        message_id=message_id,
        user_id=user_id,
        user_name=user_name,
        session_id=session_id,
        conversation_id=conversation_id,
        answer=encrypt_chat_text(answer),
        topic_id=topic_id,
    )
    if row:
        return row
    logger.warning("Chat answer update missed; creating a completed row instead.")
    row = insert_chat_message(
        user_id=user_id,
        user_name=user_name,
        session_id=session_id,
        conversation_id=conversation_id,
        question=encrypt_chat_text(question),
        answer=encrypt_chat_text(answer),
        topic_id=topic_id,
    )
    return row
