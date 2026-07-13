from __future__ import annotations

import re
import unicodedata
from typing import Any
from uuid import UUID

import httpx
from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.query import N8nQueryRequest, QueryRequest, QueryResponse
from app.services.chat import list_recent_conversation_topic_records
from app.services.chat.topic_selection import consume_topic_selection


NO_RECENT_CONVERSATION = "无近期对话"
TOPIC_ID_KEYS = ("topic_id", "current_topic_id", "selected_topic_id")

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    response_model=QueryResponse,
    response_model_exclude_none=True,
)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    return await ask_knowledge_base(request)


async def ask_knowledge_base(request: QueryRequest) -> QueryResponse:
    """Forward a user question to the n8n QA agent and normalize its answer."""

    sanitized_question = sanitize_question_for_n8n(request.question)
    n8n_request = request.model_copy(update={"question": sanitized_question})
    topic_context = await build_topic_context_for_n8n(
        user_id=request.user_id,
        session_id=request.session_id,
    )
    payload = N8nQueryRequest(
        **n8n_request.model_dump(),
        **topic_context,
    ).model_dump()
    print(
        "[query] n8n request "
        f"url={settings.n8n_query_webhook_url!r} "
        f"question={sanitized_question!r} "
        f"question_sanitized={sanitized_question != request.question!r} "
        f"user_id={request.user_id!r} "
        f"session_id={request.session_id!r} "
        f"conversation_id={request.conversation_id!r} "
        f"current_topic={topic_context['current_topic']!r} "
        f"current_summary={topic_context['current_summary']!r} "
        f"history_topics={len(topic_context['history_topics'])} topics "
        f"metadata_keys={list(request.metadata.keys())}",
        flush=True,
    )
    timeout = httpx.Timeout(
        timeout=settings.n8n_query_timeout,
        connect=settings.n8n_query_connect_timeout,
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                settings.n8n_query_webhook_url,
                json=payload,
            )
            print(
                "[query] n8n response "
                f"status={response.status_code} "
                f"body_preview={_response_text(response)!r}",
                flush=True,
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="n8n query webhook timed out") from exc
    except httpx.HTTPStatusError as exc:
        detail = _response_text(exc.response)
        raise HTTPException(
            status_code=502,
            detail=f"n8n query webhook returned {exc.response.status_code}: {detail}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to call n8n query webhook: {exc}",
        ) from exc

    raw_response = _parse_n8n_response(response)
    answer = _extract_answer(raw_response)
    topic_id = _extract_topic_id(raw_response) or consume_topic_selection(
        user_id=request.user_id,
        session_id=request.session_id,
    )

    if not answer:
        raise HTTPException(
            status_code=502,
            detail=_missing_answer_detail(raw_response),
        )

    return QueryResponse(
        question=request.question,
        answer=answer,
        topic_id=topic_id,
    )


async def build_topic_context_for_n8n(
    *,
    user_id: str | None,
    session_id: str | None,
) -> dict[str, Any]:
    """Build the compact topic-context payload n8n uses for topic routing."""

    fallback = {
        "current_topic": NO_RECENT_CONVERSATION,
        "current_summary": NO_RECENT_CONVERSATION,
        "history_topics": [],
    }
    if not user_id or not session_id:
        return fallback

    try:
        topics = await list_recent_conversation_topic_records(
            user_id=user_id,
            session_id=session_id,
            limit=settings.conversation_topic_recent_limit,
        )
    except Exception as exc:  # noqa: BLE001 - topic context should not block QA.
        print(
            "[query][warn] failed to load recent conversation topics "
            f"user_id={user_id!r} session_id={session_id!r} error={exc}",
            flush=True,
        )
        return fallback

    if not topics:
        return fallback

    history_topics = [
        {
            "topic_id": str(topic["topic_id"]),
            "topic": topic["topic"],
            "summary": topic["summary"],
        }
        for topic in topics
    ]
    return {
        "current_topic": history_topics[0]["topic"] or NO_RECENT_CONVERSATION,
        "current_summary": history_topics[0]["summary"] or NO_RECENT_CONVERSATION,
        "history_topics": history_topics,
    }


def sanitize_question_for_n8n(question: str) -> str:
    """Normalize user text before n8n inserts it into workflow JSON expressions."""

    text = unicodedata.normalize("NFC", question)
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")

    # Some clients or intermediate nodes send literal JSON escapes in plain text.
    # Collapsing these keeps the natural question while avoiding n8n hand-built JSON failures.
    replacements = {
        r"\"": '"',
        r"\'": "'",
        r"\/": "/",
        r"\n": " ",
        r"\r": " ",
        r"\t": " ",
        r"\b": " ",
        r"\f": " ",
    }
    for escaped, replacement in replacements.items():
        text = text.replace(escaped, replacement)

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t\r\n]+", " ", text)
    return text.strip()


def _parse_n8n_response(response: httpx.Response) -> Any:
    if not response.content:
        return None
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type.lower():
        return response.json()
    return response.text


def _extract_answer(raw_response: Any) -> str:
    if isinstance(raw_response, str):
        return raw_response.strip()

    if isinstance(raw_response, list):
        for item in raw_response:
            answer = _extract_answer(item)
            if answer:
                return answer
        return ""

    if isinstance(raw_response, dict):
        for key in ("answer", "output", "text", "message", "result", "data"):
            value = raw_response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, (dict, list)):
                answer = _extract_answer(value)
                if answer:
                    return answer

    return ""


def _extract_topic_id(raw_response: Any) -> str | None:
    if isinstance(raw_response, list):
        for item in raw_response:
            topic_id = _extract_topic_id(item)
            if topic_id:
                return topic_id
        return None

    if not isinstance(raw_response, dict):
        return None

    for key in TOPIC_ID_KEYS:
        topic_id = _normalize_topic_id(raw_response.get(key))
        if topic_id:
            return topic_id

    for key in ("topic", "selected_topic", "current_topic", "data", "result"):
        value = raw_response.get(key)
        if isinstance(value, (dict, list)):
            topic_id = _extract_topic_id(value)
            if topic_id:
                return topic_id
    return None


def _normalize_topic_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(UUID(text))
    except ValueError:
        return None


def _response_text(response: httpx.Response) -> str:
    text = response.text.strip()
    return text[:2000] if text else "<empty response>"


def _missing_answer_detail(raw_response: Any) -> str:
    preview = str(raw_response)
    if len(preview) > 500:
        preview = f"{preview[:500]}..."
    return (
        "n8n query webhook response did not contain an answer. "
        "Expected one of: answer, output, text, message, result, data. "
        f"Response preview: {preview}"
    )
