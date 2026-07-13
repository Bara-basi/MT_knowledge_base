from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from app.db.postgres import (
    get_conversation_topic,
    list_chat_messages_by_topic,
    update_conversation_topic,
)
from app.services.llm import LLMAPIError, LLMConfigError, get_llm_client
from app.services.privacy import decrypt_chat_text, encrypt_chat_text


logger = logging.getLogger(__name__)
SUMMARY_INTERVAL = 10
SUMMARY_MESSAGE_LIMIT = 10


def maybe_refresh_topic_summary(
    *,
    topic_id: UUID | str | None,
    user_id: str,
    session_id: str,
    message_count: int | None,
) -> dict[str, Any] | None:
    """Refresh topic summary every SUMMARY_INTERVAL completed turns."""

    if not topic_id or not message_count:
        return None
    if message_count % SUMMARY_INTERVAL != 0:
        return None
    return refresh_topic_summary(
        topic_id=topic_id,
        user_id=user_id,
        session_id=session_id,
    )


def refresh_topic_summary(
    *,
    topic_id: UUID | str,
    user_id: str,
    session_id: str,
) -> dict[str, Any] | None:
    topic = get_conversation_topic(
        topic_id=topic_id,
        user_id=user_id,
        session_id=session_id,
    )
    if not topic:
        return None

    messages = list_chat_messages_by_topic(
        topic_id=topic_id,
        user_id=user_id,
        session_id=session_id,
        limit=SUMMARY_MESSAGE_LIMIT,
    )
    if not messages:
        return None

    previous_summary = decrypt_chat_text(str(topic.get("summary") or ""))
    turns = [_decrypt_message_turn(row) for row in messages]
    new_summary = _generate_topic_summary(
        topic=decrypt_chat_text(str(topic.get("topic") or "")),
        previous_summary=previous_summary,
        turns=turns,
    )
    if not new_summary:
        return None

    updated = update_conversation_topic(
        topic_id=topic_id,
        summary=encrypt_chat_text(new_summary),
        message_increment=0,
        metadata={"summary_refreshed_at_count": int(topic.get("message_count") or 0)},
    )
    updated = dict(updated or {})
    if updated:
        updated["topic"] = decrypt_chat_text(str(updated.get("topic") or ""))
        updated["summary"] = decrypt_chat_text(str(updated.get("summary") or ""))
    return updated


def _decrypt_message_turn(row: dict[str, Any]) -> dict[str, str]:
    return {
        "question": decrypt_chat_text(str(row.get("question") or "")),
        "answer": decrypt_chat_text(str(row.get("answer") or "")),
    }


def _generate_topic_summary(
    *,
    topic: str,
    previous_summary: str,
    turns: list[dict[str, str]],
) -> str:
    transcript = "\n\n".join(
        f"用户：{turn['question']}\n助手：{turn['answer']}"
        for turn in turns
    )
    prompt = (
        "请基于上一版话题总结和最近10轮问答，生成新的中文话题总结。\n"
        "要求：\n"
        "1. 只保留对后续多轮对话有用的信息。\n"
        "2. 不要编造事实。\n"
        "3. 控制在200字以内。\n"
        "4. 直接输出总结文本，不要输出标题、JSON或解释。\n\n"
        f"话题：{topic or '未命名话题'}\n\n"
        f"上一版总结：{previous_summary or '无'}\n\n"
        f"最近10轮问答：\n{transcript}"
    )
    try:
        return get_llm_client().chat(
            [
                {
                    "role": "system",
                    "content": "你是企业内部知识库的多轮对话总结助手。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        ).strip()
    except (LLMConfigError, LLMAPIError) as exc:
        logger.warning("Failed to refresh topic summary via LLM: %s", exc)
        return ""
