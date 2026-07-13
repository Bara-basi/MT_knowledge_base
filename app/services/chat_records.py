from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import logging
from typing import Any

from app.db.postgres import (
    create_chat_message,
    ensure_chat_messages_table,
    insert_chat_message,
    touch_conversation_topic_activity,
    update_chat_answer,
)
from app.services.privacy import encrypt_chat_text


logger = logging.getLogger(__name__)
_summary_executor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="topic-summary",
)


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
        user_id=ids["user_id"],
        user_name=normalize_optional_text(user_name),
        session_id=ids["session_id"],
        conversation_id=ids["conversation_id"],
        question=question,
        answer=answer,
        topic_id=normalize_optional_text(topic_id),
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
    session_id: str,
    conversation_id: str,
    question: str,
) -> dict[str, Any]:
    ensure_chat_messages_table()
    return create_chat_message(
        user_id=user_id,
        user_name=user_name,
        session_id=session_id,
        conversation_id=conversation_id,
        question=encrypt_chat_text(question),
    )


def _record_chat_answer_sync(
    *,
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
        user_id=user_id,
        user_name=user_name,
        session_id=session_id,
        conversation_id=conversation_id,
        answer=encrypt_chat_text(answer),
        topic_id=topic_id,
    )
    if row:
        _touch_topic_activity(
            topic_id=topic_id,
            user_id=user_id,
            session_id=session_id,
        )
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
    _touch_topic_activity(
        topic_id=topic_id,
        user_id=user_id,
        session_id=session_id,
    )
    return row


def _touch_topic_activity(
    *,
    topic_id: str | None,
    user_id: str,
    session_id: str,
) -> None:
    if not topic_id:
        return
    try:
        topic = touch_conversation_topic_activity(
            topic_id=topic_id,
            user_id=user_id,
            session_id=session_id,
        )
        _schedule_topic_summary_refresh(
            topic_id=topic_id,
            user_id=user_id,
            session_id=session_id,
            message_count=int(topic.get("message_count") or 0) if topic else None,
        )
    except Exception as exc:  # noqa: BLE001 - chat persistence already succeeded.
        logger.warning("Failed to touch conversation topic activity: %s", exc)


def _schedule_topic_summary_refresh(
    *,
    topic_id: str | None,
    user_id: str,
    session_id: str,
    message_count: int | None,
) -> None:
    if not topic_id or not message_count or message_count % 10 != 0:
        return
    future = _summary_executor.submit(
        _maybe_refresh_topic_summary,
        topic_id=topic_id,
        user_id=user_id,
        session_id=session_id,
        message_count=message_count,
    )
    future.add_done_callback(_log_summary_refresh_result)


def _maybe_refresh_topic_summary(
    *,
    topic_id: str | None,
    user_id: str,
    session_id: str,
    message_count: int | None,
) -> None:
    from app.services.chat.summary import maybe_refresh_topic_summary

    maybe_refresh_topic_summary(
        topic_id=topic_id,
        user_id=user_id,
        session_id=session_id,
        message_count=message_count,
    )


def _log_summary_refresh_result(future: Future) -> None:
    try:
        future.result()
    except Exception as exc:  # noqa: BLE001 - background summarization must not affect replies.
        logger.warning("Background topic summary refresh failed: %s", exc)
