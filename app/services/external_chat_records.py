from __future__ import annotations

import asyncio
from hashlib import sha256
from typing import Any
from uuid import UUID

from app.db.postgres import (
    EXTERNAL_CHAT_ID_PREFIX,
    create_external_chat_message,
    ensure_external_chat_messages_table,
    update_external_chat_answer,
)
from app.services.privacy import encrypt_chat_text



def canonical_external_identity(*, service_id: str, value: str, kind: str) -> str:
    """Build a stable, service-scoped pseudonym that fits topic VARCHAR fields."""

    if kind not in {"user", "session"}:
        raise ValueError("external identity kind must be 'user' or 'session'")
    digest = sha256(f"{service_id}\0{kind}\0{value}".encode("utf-8")).hexdigest()
    marker = "u" if kind == "user" else "s"
    return f"{EXTERNAL_CHAT_ID_PREFIX}{marker}:{digest}"


def canonical_external_ids(
    *,
    service_id: str,
    user_id: str,
    session_id: str,
) -> dict[str, str]:
    return {
        "user_id": canonical_external_identity(
            service_id=service_id,
            value=user_id,
            kind="user",
        ),
        "session_id": canonical_external_identity(
            service_id=service_id,
            value=session_id,
            kind="session",
        ),
    }


async def create_external_chat_record(
    *,
    service_id: str,
    user_id: str,
    session_id: str,
    question: str,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _create_external_chat_record_sync,
        service_id=service_id,
        user_id=user_id,
        session_id=session_id,
        question=question,
    )


async def record_external_chat_answer(
    *,
    message_id: UUID | str,
    service_id: str,
    user_id: str,
    session_id: str,
    answer: str,
    topic_id: str | None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _record_external_chat_answer_sync,
        message_id=message_id,
        service_id=service_id,
        user_id=user_id,
        session_id=session_id,
        answer=answer,
        topic_id=topic_id,
    )


def _create_external_chat_record_sync(
    *,
    service_id: str,
    user_id: str,
    session_id: str,
    question: str,
) -> dict[str, Any]:
    ensure_external_chat_messages_table()
    return create_external_chat_message(
        service_id=service_id,
        user_id=user_id,
        session_id=session_id,
        conversation_id=session_id,
        question=encrypt_chat_text(question),
    )


def _record_external_chat_answer_sync(
    *,
    message_id: UUID | str,
    service_id: str,
    user_id: str,
    session_id: str,
    answer: str,
    topic_id: str | None,
) -> dict[str, Any]:
    row = update_external_chat_answer(
        message_id=message_id,
        service_id=service_id,
        answer=encrypt_chat_text(answer),
        topic_id=topic_id,
    )
    if not row:
        raise RuntimeError("external chat record was not found or was already completed")
    return row
