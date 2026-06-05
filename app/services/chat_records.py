from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from app.db.postgres import (
    create_chat_message,
    ensure_chat_messages_table,
    insert_chat_message,
    update_chat_answer,
    update_chat_fallback,
)


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
        session_id=ids["session_id"],
        conversation_id=ids["conversation_id"],
        question=question,
    )


async def record_chat_answer(
    *,
    user_id: str | None,
    session_id: str | None,
    conversation_id: str | None,
    question: str,
    answer: str,
) -> dict[str, Any]:
    """Persist the workflow answer to the existing user question row."""

    ids = normalize_record_ids(
        user_id=user_id,
        session_id=session_id,
        conversation_id=conversation_id,
    )
    return await asyncio.to_thread(
        _record_chat_answer_sync,
        user_id=ids["user_id"],
        session_id=ids["session_id"],
        conversation_id=ids["conversation_id"],
        question=question,
        answer=answer,
    )


async def record_chat_fallback(
    *,
    user_id: str | None,
    session_id: str | None,
    conversation_id: str | None,
    question: str,
    fallback: bool,
    reason: str | dict[str, Any] | list[Any] | None = None,
) -> dict[str, Any]:
    """Persist fallback evaluation to the existing user question row."""

    ids = normalize_record_ids(
        user_id=user_id,
        session_id=session_id,
        conversation_id=conversation_id,
    )
    return await asyncio.to_thread(
        _record_chat_fallback_sync,
        user_id=ids["user_id"],
        session_id=ids["session_id"],
        conversation_id=ids["conversation_id"],
        question=question,
        fallback=fallback,
        reason=serialize_reason(reason),
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


def serialize_reason(reason: str | dict[str, Any] | list[Any] | None) -> str:
    if reason is None:
        return ""
    if isinstance(reason, str):
        return reason.strip()
    return json.dumps(reason, ensure_ascii=False, default=str)


def _create_chat_record_sync(
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    question: str,
) -> dict[str, Any]:
    ensure_chat_messages_table()
    return create_chat_message(
        user_id=user_id,
        session_id=session_id,
        conversation_id=conversation_id,
        question=question,
    )


def _record_chat_answer_sync(
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    question: str,
    answer: str,
) -> dict[str, Any]:
    ensure_chat_messages_table()
    row = update_chat_answer(
        user_id=user_id,
        session_id=session_id,
        conversation_id=conversation_id,
        question=question,
        answer=answer,
    )
    if row:
        return row
    logger.warning("Chat answer update missed; creating a completed row instead.")
    return insert_chat_message(
        user_id=user_id,
        session_id=session_id,
        conversation_id=conversation_id,
        question=question,
        answer=answer,
        fallback=False,
        reason="",
    )


def _record_chat_fallback_sync(
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    question: str,
    fallback: bool,
    reason: str,
) -> dict[str, Any]:
    ensure_chat_messages_table()
    return update_chat_fallback(
        user_id=user_id,
        session_id=session_id,
        conversation_id=conversation_id,
        question=question,
        fallback=fallback,
        reason=reason,
    )
