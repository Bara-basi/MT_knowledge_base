from __future__ import annotations

import asyncio
import json
from typing import Any

from app.core.config import settings
from app.db.postgres import enqueue_feishu_answer_job
from app.services.privacy import encrypt_chat_text


async def enqueue_answer_job(
    *,
    dedupe_key: str,
    user_id: str,
    source_session_id: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Persist an encrypted Feishu task without placing message text in logs."""

    encrypted_payload = encrypt_chat_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return await asyncio.to_thread(
        enqueue_feishu_answer_job,
        dedupe_key=dedupe_key,
        user_id=user_id,
        source_session_id=source_session_id,
        payload=encrypted_payload,
        max_attempts=settings.answer_job_max_attempts,
    )
