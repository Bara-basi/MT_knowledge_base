from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import mimetypes
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

try:
    from PIL import Image
except Exception:  # pragma: no cover - Pillow is optional at runtime.
    Image = None

from app.api.v1.query import ask_knowledge_base
from app.core.config import settings
from app.db.minio import parse_raw_document_reference
from app.db.postgres import (
    delete_expired_feishu_answer_feedback_states,
    get_feishu_answer_feedback_state,
    insert_feishu_answer_feedback_state,
    update_feishu_answer_feedback_selection,
)
from app.schemas.query import QueryRequest
from app.services.chat_records import create_chat_record, record_chat_answer
from app.services.llm import LLMAPIError, LLMConfigError, LLMClient, LLMSettings
from app.services.privacy import decrypt_chat_text, encrypt_chat_text


router = APIRouter(prefix="/feishu", tags=["feishu"])
logger = logging.getLogger(__name__)

_tenant_access_token: str | None = None
_tenant_access_token_expires_at = 0.0
_seen_message_keys: dict[str, float] = {}
_seen_message_ttl_seconds = 3600
_pseudo_tag_pattern = re.compile(r"<(img|link)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_reference_tag_pattern = re.compile(r"<reference\b[^>]*>(.*?)</reference>", re.IGNORECASE | re.DOTALL)
_markdown_link_pattern = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)")
_markdown_link_title_pattern = re.compile(r"^(?P<url>.+?)\s+\"(?P<title>[^\"]*)\"\s*$")
_latex_block_pattern = re.compile(r"\\\[(?P<expr>.+?)\\\]", re.DOTALL)
_latex_paren_pattern = re.compile(r"\\\((?P<expr>.+?)\\\)", re.DOTALL)
_latex_dollar_pattern = re.compile(r"(?<!\\)\$(?P<expr>[^$\n]+?)(?<!\\)\$")
_latex_standalone_bracket_block_pattern = re.compile(
    r"(?m)^[ \t]*\[[ \t]*\r?\n(?P<expr>.*?)[ \t]*\r?\n[ \t]*\][ \t]*$",
    re.DOTALL,
)
_latex_bare_bracket_pattern = re.compile(
    r"(?<![!\[])\[(?!\[)(?P<expr>[^\]\n]*(?:\\[A-Za-z]+|\\\s|[_^{}])[^\]\n]*)\](?!\()"
)
_reference_link_label_pattern = re.compile(r"^\s*(片段\s*\d+)\s*[,，:：]\s*(.+?)\s*$")
_short_reference_link_pattern = r"\[\[\d+\]\]\([^)]+\)"
_adjacent_any_reference_separator_pattern = re.compile(
    rf"(?P<ref>{_short_reference_link_pattern})[ \t\r\n]+(?=(?:{_short_reference_link_pattern}))"
)
_adjacent_reference_pattern = re.compile(
    rf"(?P<ref>{_short_reference_link_pattern})(?P<separator>\s*)(?P=ref)"
)
_source_section_heading_pattern = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?(知识来源|引用文献|参考来源)(?:\*\*)?\s*[:：]?\s*$"
)
_feedback_interval_seconds = 1.5
_comfort_feedback_interval_seconds = 60.0
_feedback_texts_cache: list[str] | None = None
_local_to_lark_mapping_dir = Path("data") / "metadata" / "local2lark_mapping"
_local_to_lark_mapping_cache: dict[str, str] | None = None


@dataclass
class _ReferenceSource:
    number: int
    label: str
    description: str
    url: str


@dataclass
class _PendingImage:
    image_key: str
    alt_text: str
    size: tuple[int, int] | None


@dataclass
class _FeishuFeedbackHandle:
    state: _FeishuFeedbackState
    stop_event: asyncio.Event
    task: asyncio.Task[None]
    comfort_task: asyncio.Task[None]

    @property
    def message_id(self) -> str:
        return self.state.message_id


@dataclass
class _FeishuFeedbackState:
    message_id: str
    question: str
    prefix_text: str
    status_text: str
    update_lock: asyncio.Lock
    cancel_id: str | None = None
    canceled_text: str | None = None


@dataclass
class _FeishuAnswerCancelState:
    cancel_id: str
    started_at: float
    incoming_message_id: str | None
    chat_id: str | None
    question: str
    answer_task: asyncio.Task[None] | None = None
    feedback_handle: _FeishuFeedbackHandle | None = None
    canceled_at: float | None = None


@dataclass
class _FeishuAnswerFeedbackState:
    feedback_id: str
    answer: str
    selected: str | None
    created_at: float


_feishu_answer_cancel_states: dict[str, _FeishuAnswerCancelState] = {}
_feishu_answer_feedback_states: dict[str, _FeishuAnswerFeedbackState] = {}

_n8n_progress_texts = {
    "understanding": "🤔 正在理解问题...",
    "retrieving": "🔍 正在检索知识库...",
    "reranking": "📚 正在整理资料...",
    "generating": "✍️ 正在组织答案...",
}
_n8n_progress_stage_order = ("understanding", "retrieving", "reranking", "generating")
_n8n_optimistic_progress_thresholds = (
    (10.0, "generating"),
    (5.0, "reranking"),
    (1.5, "retrieving"),
)

# n8n execution data is only reliable after a node appears in runData.
# Node names are only labels in n8n and can be duplicated or renamed, so progress
# detection is intentionally based on authoritative node ids from the workflow.
_n8n_completed_node_stage_ids = {
    "understanding": {
        "e0e954af-c14c-47d1-917a-bdbc69eba580",
    },
    "retrieving": {
        "bdb2ff5c-c9b8-40f1-b623-dc8b912c0600",
    },
    "reranking": {
        "53fb4814-a531-4422-b6ba-f38c3db5a9a4",
        "18cbcf26-ca22-443f-aa09-3c4173e60983",
        "a509efe4-649b-4cf7-b32a-5ad35a45a60a",
        "8a620c5f-620a-481a-96f1-880f595ebb74",
    },
    "generating": {
        "6113024b-7803-4ce2-930a-c893ff1ff7fd",
    },
}

@router.post("/events")
async def handle_feishu_events(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    _debug("webhook received", client=str(request.client))
    payload = await request.json()
    _debug(
        "payload parsed",
        keys=list(payload.keys()),
        type=payload.get("type"),
        event_type=(payload.get("header") or {}).get("event_type"),
        challenge=bool(payload.get("challenge")),
    )

    if "encrypt" in payload:
        _debug("encrypted event rejected")
        raise HTTPException(
            status_code=400,
            detail="Encrypted Feishu events are not supported yet. Disable encryption or add decrypt handling.",
        )

    challenge = payload.get("challenge")
    if challenge:
        _verify_feishu_token(payload)
        _debug("challenge response sent")
        return {"challenge": challenge}

    _verify_feishu_token(payload)

    event = payload.get("event") or {}
    header = payload.get("header") or {}
    event_type = header.get("event_type") or payload.get("type")
    event_id = header.get("event_id")
    _debug("event identified", event_type=event_type, event_id=event_id)

    if _is_feishu_card_action_event(event_type, payload):
        result = await _handle_feishu_card_action(payload)
        _debug("card action handled", event_type=event_type, result=result)
        return result

    if event_type != "im.message.receive_v1":
        _debug("event ignored", reason="unsupported_event_type", event_type=event_type)
        return {"ok": True, "ignored": True, "event_type": event_type}

    message = event.get("message") or {}
    message_id = message.get("message_id")
    dedupe_key = _build_dedupe_key(message_id, event_id)
    if _is_duplicate_message(dedupe_key):
        _debug(
            "event ignored",
            reason="duplicate_message",
            event_id=event_id,
            message_id=message_id,
            dedupe_key=dedupe_key,
        )
        return {
            "ok": True,
            "duplicate": True,
            "event_id": event_id,
            "message_id": message_id,
        }

    question = _extract_message_text(message)
    _debug(
        "message parsed",
        event_id=event_id,
        message_id=message_id,
        dedupe_key=dedupe_key,
        chat_id=message.get("chat_id"),
        chat_type=message.get("chat_type"),
        message_type=message.get("message_type"),
        question_len=len(question),
        question_preview=_preview(question),
    )

    if not question:
        _debug("event ignored", reason="unsupported_or_empty_message", message_id=message_id)
        if message_id or message.get("chat_id"):
            background_tasks.add_task(
                _send_unsupported_feishu_message_notice,
                chat_id=message.get("chat_id"),
                message_id=message_id,
            )
        return {"ok": True, "ignored": True, "reason": "unsupported_or_empty_message"}

    background_tasks.add_task(
        _answer_feishu_message,
        question=question,
        sender_id=_extract_sender_id(event.get("sender") or {}),
        sender_name=_extract_sender_name(event.get("sender") or {}),
        chat_id=message.get("chat_id"),
        message_id=message_id,
        event_id=event_id,
        chat_type=message.get("chat_type"),
        dedupe_key=dedupe_key,
    )
    _debug(
        "background task queued",
        event_id=event_id,
        message_id=message_id,
        dedupe_key=dedupe_key,
    )

    return {
        "ok": True,
        "accepted": True,
        "event_id": event_id,
        "message_id": message_id,
    }


async def _answer_feishu_message(
    *,
    question: str,
    sender_id: str | None,
    sender_name: str | None,
    chat_id: str | None,
    message_id: str | None,
    event_id: str | None,
    chat_type: str | None,
    dedupe_key: str | None,
) -> None:
    _debug(
        "background task started",
        event_id=event_id,
        message_id=message_id,
        dedupe_key=dedupe_key,
        chat_id=chat_id,
        question_len=len(question),
    )

    n8n_started_after = time.time()
    cancel_id = _register_feishu_answer_cancel_state(
        question=question,
        chat_id=chat_id,
        message_id=message_id,
    )
    feedback_handle_task = _schedule_initial_feishu_feedback(
        question=question,
        chat_id=chat_id,
        message_id=message_id,
        event_id=event_id,
        dedupe_key=dedupe_key,
        n8n_started_after=n8n_started_after,
        cancel_id=cancel_id,
    )

    try:
        await create_chat_record(
            user_id=sender_id,
            user_name=sender_name,
            session_id=chat_id,
            conversation_id=chat_id,
            question=question,
        )
        _debug(
            "chat record created",
            event_id=event_id,
            message_id=message_id,
            dedupe_key=dedupe_key,
        )
    except Exception as exc:  # noqa: BLE001 - logging must not block user replies.
        logger.exception("Failed to create chat record for Feishu message: %s", exc)
        _warn(
            "chat record create failed",
            event_id=event_id,
            message_id=message_id,
            dedupe_key=dedupe_key,
            error=str(exc),
        )

    try:
        _debug(
            "query start",
            event_id=event_id,
            message_id=message_id,
            dedupe_key=dedupe_key,
            n8n_url=settings.n8n_query_webhook_url,
            timeout=settings.n8n_query_timeout,
        )
        response = await ask_knowledge_base(
            QueryRequest(
                question=question,
                user_id=sender_id,
                session_id=chat_id,
                conversation_id=chat_id,
                metadata={
                    "source": "feishu",
                    "event_id": event_id,
                    "message_id": message_id,
                    "dedupe_key": dedupe_key,
                    "chat_type": chat_type,
                },
            )
        )
        _debug(
            "query success",
            event_id=event_id,
            message_id=message_id,
            dedupe_key=dedupe_key,
            answer_len=len(response.answer),
            answer_preview=_preview(response.answer),
        )
        if _is_transient_feedback_answer(response.answer):
            _debug(
                "query returned transient feedback; final reply skipped",
                event_id=event_id,
                message_id=message_id,
                dedupe_key=dedupe_key,
                answer_preview=_preview(response.answer),
            )
            feedback_handle = await _resolve_initial_feishu_feedback(
                feedback_handle_task,
                wait=True,
            )
            await _stop_feishu_feedback_loop(feedback_handle)
            _discard_feishu_answer_cancel_state(cancel_id)
            return

        if _is_feishu_answer_cancelled(cancel_id):
            _debug(
                "query result ignored",
                reason="answer_cancelled",
                event_id=event_id,
                message_id=message_id,
                dedupe_key=dedupe_key,
                cancel_id=cancel_id,
            )
            feedback_handle = await _resolve_initial_feishu_feedback(
                feedback_handle_task,
                wait=False,
            )
            await _stop_feishu_feedback_loop(feedback_handle)
            _discard_feishu_answer_cancel_state(cancel_id)
            return

        try:
            await record_chat_answer(
                user_id=sender_id,
                user_name=sender_name,
                session_id=chat_id,
                conversation_id=chat_id,
                question=question,
                answer=response.answer,
            )
            _debug(
                "chat answer recorded",
                event_id=event_id,
                message_id=message_id,
                dedupe_key=dedupe_key,
            )
        except Exception as exc:  # noqa: BLE001 - logging must not block user replies.
            logger.exception("Failed to record chat answer for Feishu message: %s", exc)
            _warn(
                "chat answer record failed",
                event_id=event_id,
                message_id=message_id,
                dedupe_key=dedupe_key,
                error=str(exc),
            )
    except asyncio.CancelledError:
        _debug(
            "answer task cancelled",
            event_id=event_id,
            message_id=message_id,
            dedupe_key=dedupe_key,
            cancel_id=cancel_id,
        )
        feedback_handle = await _resolve_initial_feishu_feedback(feedback_handle_task, wait=False)
        await _stop_feishu_feedback_loop(feedback_handle)
        _discard_feishu_answer_cancel_state(cancel_id)
        return
    except HTTPException as exc:
        detail = _detail_to_text(exc.detail)
        logger.exception("Failed to query knowledge base for Feishu message: %s", detail)
        _debug(
            "query failed",
            event_id=event_id,
            message_id=message_id,
            dedupe_key=dedupe_key,
            error=detail,
        )
        feedback_handle = await _resolve_initial_feishu_feedback(feedback_handle_task, wait=False)
        await _stop_feishu_feedback_loop(feedback_handle)
        if _is_feishu_answer_cancelled(cancel_id):
            _debug(
                "query failure ignored",
                reason="answer_cancelled",
                event_id=event_id,
                message_id=message_id,
                dedupe_key=dedupe_key,
                cancel_id=cancel_id,
            )
            _discard_feishu_answer_cancel_state(cancel_id)
            return
        failed_text = _get_failed_feedback_text()
        if feedback_handle:
            await _try_update_feishu_markdown_message(feedback_handle.message_id, failed_text)
        elif message_id:
            await _try_reply_feishu_markdown(message_id, failed_text)
        elif chat_id:
            await _try_send_feishu_markdown_message(
                chat_id=chat_id,
                reply_to_message_id=None,
                markdown_text=failed_text,
            )
        _discard_feishu_answer_cancel_state(cancel_id)
        return

    feedback_handle = await _resolve_initial_feishu_feedback(feedback_handle_task, wait=False)
    await _stop_feishu_feedback_loop(feedback_handle)

    if _is_feishu_answer_cancelled(cancel_id):
        _debug(
            "reply skipped",
            reason="answer_cancelled",
            event_id=event_id,
            message_id=message_id,
            dedupe_key=dedupe_key,
            cancel_id=cancel_id,
        )
        _discard_feishu_answer_cancel_state(cancel_id)
        return

    if not message_id:
        _debug(
            "reply skipped",
            reason="missing_message_id",
            event_id=event_id,
            dedupe_key=dedupe_key,
        )
        _discard_feishu_answer_cancel_state(cancel_id)
        return

    answer_feedback_id = await _register_feishu_answer_feedback_state(response.answer)
    if feedback_handle:
        sent = await _try_update_feishu_markdown_message(
            feedback_handle.message_id,
            response.answer,
            answer_feedback_id=answer_feedback_id,
        )
    else:
        sent_message_id = await _try_send_feishu_markdown_message(
            chat_id=chat_id,
            reply_to_message_id=message_id,
            markdown_text=response.answer,
            answer_feedback_id=answer_feedback_id,
        )
        sent = sent_message_id is not None
    _debug(
        "reply finished",
        event_id=event_id,
        message_id=message_id,
        dedupe_key=dedupe_key,
        sent=sent,
    )
    _discard_feishu_answer_cancel_state(cancel_id)


def _schedule_initial_feishu_feedback(
    *,
    question: str,
    chat_id: str | None,
    message_id: str | None,
    event_id: str | None,
    dedupe_key: str | None,
    n8n_started_after: float,
    cancel_id: str | None = None,
) -> asyncio.Task[_FeishuFeedbackHandle | None] | None:
    if not message_id and not chat_id:
        _debug(
            "initial feedback skipped",
            reason="missing_message_and_chat_id",
            event_id=event_id,
            dedupe_key=dedupe_key,
        )
        return None

    task = asyncio.create_task(
        _send_initial_feishu_feedback(
            question=question,
            chat_id=chat_id,
            message_id=message_id,
            event_id=event_id,
            dedupe_key=dedupe_key,
            n8n_started_after=n8n_started_after,
            cancel_id=cancel_id,
        )
    )
    task.add_done_callback(_log_initial_feishu_feedback_task_failure)
    return task


async def _resolve_initial_feishu_feedback(
    task: asyncio.Task[_FeishuFeedbackHandle | None] | None,
    *,
    wait: bool = True,
) -> _FeishuFeedbackHandle | None:
    if not task:
        return None
    if not wait and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return None
    try:
        return await task
    except asyncio.CancelledError:
        return None
    except Exception as exc:  # noqa: BLE001 - feedback cannot block the final answer.
        logger.exception("Unexpected Feishu initial feedback task failure: %s", exc)
        return None


def _log_initial_feishu_feedback_task_failure(
    task: asyncio.Task[_FeishuFeedbackHandle | None],
) -> None:
    if task.cancelled():
        return
    try:
        task.result()
    except Exception as exc:  # noqa: BLE001 - keep background task failures visible.
        logger.exception("Unexpected Feishu initial feedback task failure: %s", exc)


async def _send_initial_feishu_feedback(
    *,
    question: str,
    chat_id: str | None,
    message_id: str | None,
    event_id: str | None,
    dedupe_key: str | None,
    n8n_started_after: float,
    cancel_id: str | None = None,
) -> _FeishuFeedbackHandle | None:
    initial_feedback = await _send_initial_feedback_card_with_greeting(
        question=question,
        chat_id=chat_id,
        message_id=message_id,
        cancel_id=cancel_id,
    )
    if initial_feedback is None:
        _debug(
            "initial feedback skipped",
            reason="greeting_model_returned_null",
            event_id=event_id,
            message_id=message_id,
            dedupe_key=dedupe_key,
        )
        return None
    feedback_message_id, prefix_text, status_text = initial_feedback

    if not feedback_message_id:
        _debug(
            "initial feedback send skipped",
            reason="send_failed",
            event_id=event_id,
            message_id=message_id,
            dedupe_key=dedupe_key,
        )
        return None

    stop_event = asyncio.Event()
    feedback_state = _FeishuFeedbackState(
        message_id=feedback_message_id,
        question=question,
        prefix_text=prefix_text,
        status_text=status_text,
        update_lock=asyncio.Lock(),
        cancel_id=cancel_id,
    )
    feedback_task = _create_feishu_feedback_task(
        state=feedback_state,
        stop_event=stop_event,
        event_id=event_id,
        dedupe_key=dedupe_key,
        n8n_started_after=n8n_started_after,
    )
    comfort_task = asyncio.create_task(
        _run_feishu_comfort_feedback_loop(
            state=feedback_state,
            stop_event=stop_event,
            event_id=event_id,
            dedupe_key=dedupe_key,
        )
    )
    _debug(
        "feedback card started",
        event_id=event_id,
        message_id=message_id,
        feedback_message_id=feedback_message_id,
        dedupe_key=dedupe_key,
    )
    handle = _FeishuFeedbackHandle(
        state=feedback_state,
        stop_event=stop_event,
        task=feedback_task,
        comfort_task=comfort_task,
    )
    _bind_feishu_answer_feedback(cancel_id, handle)
    return handle


async def _send_initial_feedback_card_with_greeting(
    *,
    question: str,
    chat_id: str | None,
    message_id: str | None,
    cancel_id: str | None = None,
) -> tuple[str | None, str, str] | None:
    initial_parts = await _build_initial_feedback_parts(question)
    if initial_parts is None:
        return None
    prefix_text, status_text = initial_parts
    feedback_message_id = await _send_initial_feedback_card(
        chat_id=chat_id,
        message_id=message_id,
        markdown_text=_compose_feedback_card_text(prefix_text, status_text),
        cancel_id=cancel_id,
    )
    return feedback_message_id, prefix_text, status_text


async def _send_initial_feedback_card(
    *,
    chat_id: str | None,
    message_id: str | None,
    markdown_text: str,
    cancel_id: str | None = None,
) -> str | None:
    if message_id:
        feedback_message_id = await _try_send_feishu_markdown_message(
            chat_id=chat_id,
            reply_to_message_id=message_id,
            markdown_text=markdown_text,
            log_content=False,
            stop_cancel_id=cancel_id,
        )
        if feedback_message_id:
            return feedback_message_id
        return await _try_send_feishu_markdown_reply(
            message_id,
            markdown_text,
            log_content=False,
            stop_cancel_id=cancel_id,
        )
    if chat_id:
        return await _try_send_feishu_markdown_message(
            chat_id=chat_id,
            reply_to_message_id=None,
            markdown_text=markdown_text,
            log_content=False,
            stop_cancel_id=cancel_id,
        )
    return None


async def _build_initial_feedback_text(question: str) -> str | None:
    parts = await _build_initial_feedback_parts(question)
    if parts is None:
        return None
    return _compose_feedback_card_text(*parts)


async def _build_initial_feedback_parts(question: str) -> tuple[str, str] | None:
    feedback_text = _get_initial_feedback_text()
    greeting_text = await _generate_immediate_feedback_greeting(question)
    if greeting_text is None:
        return None
    return greeting_text.strip(), feedback_text.strip()


async def _generate_immediate_feedback_greeting(question: str) -> str | None:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_generate_immediate_feedback_greeting_sync, question),
            timeout=max(settings.immediate_feedback_timeout, 0.1),
        )
    except (asyncio.TimeoutError, LLMAPIError, LLMConfigError) as exc:
        _warn("immediate feedback greeting fallback", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - feedback must stay best-effort.
        logger.exception("Failed to generate immediate Feishu greeting: %s", exc)
    return _get_local_immediate_feedback_greeting(question)


def _generate_immediate_feedback_greeting_sync(question: str) -> str | None:
    if not settings.siliconflow_api_key:
        raise LLMConfigError("Missing SILICONFLOW_API_KEY for immediate feedback greeting")

    llm_settings = LLMSettings(
        api_key=settings.siliconflow_api_key,
        base_url=settings.siliconflow_base_url,
        model=settings.immediate_feedback_model,
        timeout=settings.immediate_feedback_timeout,
        connect_timeout=settings.immediate_feedback_connect_timeout,
        read_timeout=settings.immediate_feedback_timeout,
        write_timeout=settings.immediate_feedback_timeout,
        pool_timeout=min(settings.immediate_feedback_timeout, 1.0),
    )
    client = LLMClient(llm_settings)
    messages = _build_immediate_feedback_messages(question)
    chat_kwargs = {
        "model": settings.immediate_feedback_model,
        "temperature": 0.3,
        "max_tokens": settings.immediate_feedback_max_tokens,
    }
    reply = _chat_immediate_feedback_greeting(
        client,
        messages,
        chat_kwargs,
        extra_body=_build_immediate_feedback_extra_body(),
    )
    return _clean_immediate_feedback_greeting(reply)


def _build_immediate_feedback_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                    "你是一个企业内部知识库助手的“快速响应模式”。"

                    "你的任务是：在用户提问后，立即生成一个自然、像真人一样的短回复，用于缓解等待焦虑。"

                    "要求："
                    "1. 不要给出正式答案"
                    "2. 不要引用制度或文档内容"
                    "3. 不要编造具体事实"
                    "4. 不要输出长段解释"
                    "5. 语气要自然、像人在帮忙查资料"
                    "6. 可以表达“好，正在查找 / 正在确认 / 我帮你看看 / 等会，我查一下资料”等状态"
                    "7. 长度控制在1~2句话"
                    "8. 如果用户只是在打招呼、提问过于简单、日常闲聊，或只是抱怨/倾诉而不需要检索知识库，请只输出 null"
                    ""
                    "你可以表达的内容包括："
                    "- 正在帮用户查找"
                    "- 正在匹配相关信息"
                    "- 很快会返回结果"
                    "- 这个问题需要稍等确认"
                    "- 评价用户的提问(结合实际，比如有趣、有点棘手、需要推理等)"
                    "- 如果用户很复杂，提示用户可能要花点时间"

                    "禁止："
                    "- 不要给结论"
                    "- 不要假装已经查完"
                    "- 不要引用具体制度条款"
                    "- 不要输出步骤答案"
                    "- 除 null 场景外，不要输出 null 以外的控制词"

            ),
        },
        {
            "role": "user",
            "content": f"用户问题：{_truncate_question_for_feedback(question)}",
        },
    ]


def _build_immediate_feedback_extra_body() -> dict[str, Any]:
    value = str(settings.immediate_feedback_enable_thinking).strip().lower()
    if value in {"", "auto"}:
        return {}
    if value in {"off", "false", "0", "no", "n"}:
        return {"enable_thinking": False}
    if value in {"on", "true", "1", "yes", "y"}:
        return {"enable_thinking": True}
    return {"enable_thinking": settings.immediate_feedback_enable_thinking}


def _chat_immediate_feedback_greeting(
    client: LLMClient,
    messages: list[dict[str, str]],
    chat_kwargs: dict[str, Any],
    *,
    extra_body: dict[str, Any],
) -> str:
    try:
        return client.chat(
            messages,
            **chat_kwargs,
            extra_body=extra_body,
        )
    except LLMAPIError:
        if not getattr(settings, "immediate_feedback_retry_without_thinking_options", True):
            raise
        return client.chat(
            messages,
            **chat_kwargs,
        )


def _truncate_question_for_feedback(question: str) -> str:
    text = re.sub(r"\s+", " ", question).strip()
    return text[:300]


def _clean_immediate_feedback_greeting(text: str) -> str | None:
    if _is_null_immediate_feedback(text):
        return None
    lines = [line.strip().strip("\"'") for line in text.splitlines() if line.strip()]
    cleaned = " ".join(lines[:2])
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if _is_null_immediate_feedback(cleaned):
        return None
    if len(cleaned) > 120:
        cleaned = f"{cleaned[:117].rstrip()}..."
    return cleaned


def _is_null_immediate_feedback(text: str) -> bool:
    normalized = text.strip().strip("`").strip().strip("\"'").strip().lower()
    return normalized in {"null", "none"}


def _get_local_immediate_feedback_greeting(question: str) -> str:
    normalized = _truncate_question_for_feedback(question)
    if len(normalized) >= 80:
        return "收到，这个问题信息量有点大，我先检索知识库再帮你整理清楚。"
    if any(keyword in normalized for keyword in ("怎么", "如何", "为什么", "哪些", "是否", "能不能")):
        return "收到，我先帮你查一下相关资料，再整理成好读的答复。"
    return "收到，我先看一下知识库里的相关内容，马上给你整理答复。"


def _compose_feedback_card_text(prefix_text: str, status_text: str) -> str:
    prefix_text = prefix_text.strip()
    status_text = status_text.strip()
    if prefix_text and status_text:
        return f"{prefix_text}\n\n{status_text}"
    return prefix_text or status_text


def _compose_cancelled_feedback_card_text(state: _FeishuFeedbackState) -> str:
    return state.prefix_text.strip() or state.question.strip() or " "


def _is_transient_feedback_answer(answer: str) -> bool:
    normalized_answer = _normalize_feedback_status_text(answer)
    if not normalized_answer:
        return False

    status_texts = set(_n8n_progress_texts.values())
    status_texts.update(_load_feedback_texts())
    status_texts.add(_get_failed_feedback_text())
    status_texts.update(
        {
            "正在理解问题",
            "正在检索知识库",
            "正在整理资料",
            "正在组织答案",
            "处理失败，请稍后重试",
        }
    )

    normalized_statuses = {
        _normalize_feedback_status_text(text)
        for text in status_texts
        if text and _normalize_feedback_status_text(text)
    }
    return normalized_answer in normalized_statuses


def _normalize_feedback_status_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", text).strip().lower()


def _create_feishu_feedback_task(
    *,
    state: _FeishuFeedbackState,
    stop_event: asyncio.Event,
    event_id: str | None,
    dedupe_key: str | None,
    n8n_started_after: float,
) -> asyncio.Task[None]:
    if _n8n_progress_polling_configured():
        return asyncio.create_task(
            _run_n8n_progress_feedback_loop(
                state=state,
                stop_event=stop_event,
                event_id=event_id,
                dedupe_key=dedupe_key,
                started_after=n8n_started_after,
            )
        )

    return asyncio.create_task(
        _run_feishu_feedback_loop(
            state=state,
            stop_event=stop_event,
            event_id=event_id,
            dedupe_key=dedupe_key,
        )
    )


async def _run_feishu_comfort_feedback_loop(
    *,
    state: _FeishuFeedbackState,
    stop_event: asyncio.Event,
    event_id: str | None,
    dedupe_key: str | None,
) -> None:
    minute = 0
    try:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=max(_comfort_feedback_interval_seconds, 0.1),
                )
                break
            except asyncio.TimeoutError:
                pass

            minute += 1
            comfort_text = await _generate_comfort_feedback_text(
                question=state.question,
                minute=minute,
                status_text=state.status_text,
            )
            if stop_event.is_set():
                break
            await _update_feishu_feedback_prefix(state, comfort_text)
    except Exception as exc:  # noqa: BLE001 - comfort feedback must not block answers.
        _warn(
            "comfort feedback loop failed",
            event_id=event_id,
            dedupe_key=dedupe_key,
            message_id=state.message_id,
            error=str(exc),
        )


async def _generate_comfort_feedback_text(
    *,
    question: str,
    minute: int,
    status_text: str,
) -> str:
    try:
        text = await asyncio.wait_for(
            asyncio.to_thread(
                _generate_comfort_feedback_text_sync,
                question,
                minute,
                status_text,
            ),
            timeout=max(settings.immediate_feedback_timeout, 0.1),
        )
        cleaned = _clean_comfort_feedback_text(text)
        if cleaned:
            return cleaned
    except (asyncio.TimeoutError, LLMAPIError, LLMConfigError) as exc:
        _warn("comfort feedback model fallback", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - comfort feedback must stay best-effort.
        logger.exception("Failed to generate comfort feedback text: %s", exc)
    return _get_local_comfort_feedback_text(minute, status_text)


def _generate_comfort_feedback_text_sync(
    question: str,
    minute: int,
    status_text: str,
) -> str:
    if not settings.siliconflow_api_key:
        raise LLMConfigError("Missing SILICONFLOW_API_KEY for comfort feedback")

    llm_settings = LLMSettings(
        api_key=settings.siliconflow_api_key,
        base_url=settings.siliconflow_base_url,
        model=settings.immediate_feedback_model,
        timeout=settings.immediate_feedback_timeout,
        connect_timeout=settings.immediate_feedback_connect_timeout,
        read_timeout=settings.immediate_feedback_timeout,
        write_timeout=settings.immediate_feedback_timeout,
        pool_timeout=min(settings.immediate_feedback_timeout, 1.0),
    )
    client = LLMClient(llm_settings)
    messages = _build_comfort_feedback_messages(
        question=question,
        minute=minute,
        status_text=status_text,
    )
    return _chat_immediate_feedback_greeting(
        client,
        messages,
        {
            "model": settings.immediate_feedback_model,
            "temperature": 0.45,
            "max_tokens": min(settings.immediate_feedback_max_tokens, 60),
        },
        extra_body=_build_immediate_feedback_extra_body(),
    )


def _build_comfort_feedback_messages(
    *,
    question: str,
    minute: int,
    status_text: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是企业知识库助手的等待安抚机器人。用户已经等了一会儿，"
                "你需要根据用户问题和当前处理状态，生成一句自然、贴近问题的安慰语。"
                "不要回答问题，不要给结论，不要编造事实，不要反问。"
                "控制在35字以内，像同事在帮忙处理。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题：{_truncate_question_for_feedback(question)}\n"
                f"已等待：约{minute}分钟\n"
                f"当前状态：{status_text}\n"
                "请只输出安慰语本身。"
            ),
        },
    ]


def _clean_comfort_feedback_text(text: str) -> str:
    lines = [line.strip().strip("\"'") for line in text.splitlines() if line.strip()]
    cleaned = " ".join(lines[:2])
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or _is_null_immediate_feedback(cleaned):
        return ""
    if len(cleaned) > 80:
        cleaned = f"{cleaned[:77].rstrip()}..."
    return cleaned


def _get_local_comfort_feedback_text(minute: int, status_text: str) -> str:
    status_hint = _status_text_to_comfort_hint(status_text)
    templates = [
        f"快好了，我还在{status_hint}，整理清楚后马上发你。",
        f"还差一点，我再{status_hint}，请稍等一下。",
        f"资料还在处理中，我继续{status_hint}，不会让你干等太久。",
        f"我还在跟进这个问题，先{status_hint}，整理好就回复你。",
    ]
    return templates[(max(minute, 1) - 1) % len(templates)]


def _status_text_to_comfort_hint(status_text: str) -> str:
    normalized = _normalize_feedback_status_text(status_text)
    if "检索" in normalized:
        return "检索知识库"
    if "整理" in normalized:
        return "整理资料"
    if "组织" in normalized or "答案" in normalized:
        return "组织答案"
    if "理解" in normalized:
        return "确认问题方向"
    return "处理相关资料"


async def _update_feishu_feedback_prefix(
    state: _FeishuFeedbackState,
    prefix_text: str,
) -> bool:
    state.prefix_text = prefix_text.strip()
    return await _update_feishu_feedback_card(state)


async def _update_feishu_feedback_status(
    state: _FeishuFeedbackState,
    status_text: str,
) -> bool:
    state.status_text = status_text.strip()
    return await _update_feishu_feedback_card(state)


async def _update_feishu_feedback_card(state: _FeishuFeedbackState) -> bool:
    async with state.update_lock:
        canceled_text = state.canceled_text
        return await _try_update_feishu_markdown_message(
            state.message_id,
            _compose_feedback_card_text(state.prefix_text, state.status_text),
            log_content=False,
            stop_cancel_id=None if canceled_text else state.cancel_id,
            canceled_text=canceled_text,
        )


def _n8n_progress_polling_configured() -> bool:
    return (
        settings.n8n_progress_enabled
        and bool(settings.n8n_api_key)
        and bool(_get_n8n_api_base_url())
    )


def _get_n8n_api_base_url() -> str:
    if settings.n8n_api_base_url:
        return settings.n8n_api_base_url

    webhook_url = settings.n8n_query_webhook_url
    marker = "/webhook"
    marker_index = webhook_url.find(marker)
    if marker_index > 0:
        return webhook_url[:marker_index].rstrip("/")
    return ""


async def _run_n8n_progress_feedback_loop(
    *,
    state: _FeishuFeedbackState,
    stop_event: asyncio.Event,
    event_id: str | None,
    dedupe_key: str | None,
    started_after: float,
) -> None:
    last_stage = "understanding"
    last_text = _n8n_progress_texts[last_stage]
    poll_interval = max(settings.n8n_progress_poll_interval, 0.2)
    loop_started_at = time.monotonic()

    try:
        while not stop_event.is_set():
            stage = await _poll_n8n_progress_stage(started_after=started_after)
            optimistic_stage = _optimistic_n8n_progress_stage(
                elapsed_seconds=time.monotonic() - loop_started_at,
            )
            next_stage = _latest_n8n_progress_stage(stage, optimistic_stage)
            if stop_event.is_set():
                break
            if next_stage and _n8n_stage_index(next_stage) > _n8n_stage_index(last_stage):
                last_stage = next_stage
                text = _n8n_progress_texts[next_stage]
                if text != last_text:
                    last_text = text
                    await _update_feishu_feedback_status(state, text)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
                break
            except asyncio.TimeoutError:
                pass
    except Exception as exc:  # noqa: BLE001 - progress feedback should never block answers.
        _warn(
            "n8n progress polling failed; falling back to legacy feedback loop",
            event_id=event_id,
            dedupe_key=dedupe_key,
            error=str(exc),
        )
        await _run_feishu_feedback_loop(
            state=state,
            stop_event=stop_event,
            event_id=event_id,
            dedupe_key=dedupe_key,
        )


async def _poll_n8n_progress_stage(*, started_after: float) -> str | None:
    base_url = _get_n8n_api_base_url()
    if not base_url:
        return None

    workflow_ids = [
        workflow_id
        for workflow_id in (
            settings.n8n_query_workflow_id,
            settings.n8n_retrieval_workflow_id,
        )
        if workflow_id
    ]
    if not workflow_ids:
        return None

    timeout = httpx.Timeout(
        timeout=min(max(settings.n8n_progress_poll_interval, 1.0), 10.0),
        connect=min(settings.n8n_query_connect_timeout, 5.0),
    )
    headers = {"X-N8N-API-KEY": settings.n8n_api_key}

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        executions: list[dict[str, Any]] = []
        for workflow_id in workflow_ids:
            for status in (None, "running", "success", "error"):
                executions.extend(
                    await _list_recent_n8n_executions(
                        client=client,
                        base_url=base_url,
                        workflow_id=workflow_id,
                        started_after=started_after,
                        status=status,
                    )
                )

        executions_by_id = {
            str(execution.get("id")): execution
            for execution in executions
            if execution.get("id")
        }

        stage: str | None = None
        for execution in sorted(
            executions_by_id.values(),
            key=lambda item: _parse_n8n_time(item.get("startedAt")) or 0.0,
        ):
            execution_id = execution.get("id")
            if not execution_id:
                continue

            detail = await _get_n8n_execution_detail(
                client=client,
                base_url=base_url,
                execution_id=str(execution_id),
            )
            detail_stage = _extract_n8n_progress_stage(detail)
            if detail_stage and (
                stage is None or _n8n_stage_index(detail_stage) > _n8n_stage_index(stage)
            ):
                stage = detail_stage

        return stage


async def _list_recent_n8n_executions(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    workflow_id: str,
    started_after: float,
    status: str | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"workflowId": workflow_id, "limit": 10, "includeData": "false"}
    if status:
        params["status"] = status

    response = await client.get(
        f"{base_url}/api/v1/executions",
        params=params,
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []

    threshold = started_after - max(settings.n8n_progress_lookback_seconds, 0)
    recent_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        started_at = _parse_n8n_time(item.get("startedAt"))
        if started_at is None or started_at >= threshold:
            recent_items.append(item)
    return recent_items


async def _get_n8n_execution_detail(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    execution_id: str,
) -> dict[str, Any]:
    response = await client.get(
        f"{base_url}/api/v1/executions/{execution_id}",
        params={"includeData": "true"},
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _extract_n8n_progress_stage(execution: dict[str, Any]) -> str | None:
    stage: str | None = None
    for node_id, node_name, _started_at in _iter_n8n_executed_nodes(execution):
        node_stage = _stage_for_n8n_node(node_id=node_id, node_name=node_name)
        if node_stage and (
            stage is None or _n8n_stage_index(node_stage) > _n8n_stage_index(stage)
        ):
            stage = node_stage
    return stage


def _iter_n8n_executed_nodes(
    execution: dict[str, Any],
) -> list[tuple[str | None, str, float | None]]:
    data = execution.get("data") if isinstance(execution.get("data"), dict) else {}
    result_data = data.get("resultData") if isinstance(data.get("resultData"), dict) else {}
    run_data = result_data.get("runData") if isinstance(result_data.get("runData"), dict) else {}
    if not run_data:
        return []

    nodes = _extract_n8n_workflow_nodes(execution)
    nodes_by_name = {
        str(node.get("name")): node
        for node in nodes
        if isinstance(node, dict) and node.get("name")
    }

    executed_nodes: list[tuple[str | None, str, float | None]] = []
    for node_name, runs in run_data.items():
        if not isinstance(node_name, str):
            continue
        node = nodes_by_name.get(node_name) or {}
        started_at = _first_n8n_node_start_time(runs)
        executed_nodes.append((node.get("id"), node_name, started_at))

    return sorted(executed_nodes, key=lambda item: item[2] or 0.0)


def _extract_n8n_workflow_nodes(execution: dict[str, Any]) -> list[dict[str, Any]]:
    execution_data = execution.get("data") if isinstance(execution.get("data"), dict) else {}
    candidates = [
        execution.get("workflowData"),
        execution_data.get("workflowData"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and isinstance(candidate.get("nodes"), list):
            return [node for node in candidate["nodes"] if isinstance(node, dict)]
    return []


def _first_n8n_node_start_time(runs: Any) -> float | None:
    if not isinstance(runs, list):
        return None
    starts = [
        _parse_n8n_time(run.get("startTime"))
        for run in runs
        if isinstance(run, dict)
    ]
    starts = [value for value in starts if value is not None]
    return min(starts) if starts else None


def _stage_for_n8n_node(*, node_id: Any, node_name: str) -> str | None:
    node_id_text = str(node_id) if node_id else ""
    for stage in _n8n_progress_stage_order:
        if node_id_text and node_id_text in _n8n_completed_node_stage_ids[stage]:
            return stage
    return None


def _optimistic_n8n_progress_stage(*, elapsed_seconds: float) -> str | None:
    for threshold, stage in _n8n_optimistic_progress_thresholds:
        if elapsed_seconds >= threshold:
            return stage
    return None


def _latest_n8n_progress_stage(*stages: str | None) -> str | None:
    latest_stage: str | None = None
    for stage in stages:
        if stage and (
            latest_stage is None or _n8n_stage_index(stage) > _n8n_stage_index(latest_stage)
        ):
            latest_stage = stage
    return latest_stage


def _n8n_stage_index(stage: str | None) -> int:
    if stage in _n8n_progress_stage_order:
        return _n8n_progress_stage_order.index(stage)
    return -1


def _parse_n8n_time(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


async def _run_feishu_feedback_loop(
    *,
    state: _FeishuFeedbackState,
    stop_event: asyncio.Event,
    event_id: str | None,
    dedupe_key: str | None,
) -> None:
    feedback_texts = _load_feedback_texts()
    if not feedback_texts:
        _debug("feedback loop skipped", reason="missing_feedback_texts", message_id=state.message_id)
        return

    index = 1 % len(feedback_texts)
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_feedback_interval_seconds)
            break
        except asyncio.TimeoutError:
            pass

        text = feedback_texts[index]
        index = (index + 1) % len(feedback_texts)
        if stop_event.is_set():
            break
        await _update_feishu_feedback_status(state, text)


async def _stop_feishu_feedback_loop(
    handle: _FeishuFeedbackHandle | None,
) -> None:
    if not handle:
        return
    handle.stop_event.set()
    await asyncio.gather(handle.task, handle.comfort_task, return_exceptions=True)


def _load_feedback_texts() -> list[str]:
    global _feedback_texts_cache

    if _feedback_texts_cache is not None:
        return _feedback_texts_cache

    fallback_texts = [
        "🤔 正在理解问题...",
        "🔍 正在检索知识库...",
        "📚 正在整理资料...",
        "✍️ 正在组织答案...",
    ]
    feedback_path = Path("data") / "UI反馈.json"
    try:
        raw_data = json.loads(feedback_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _warn("feedback text load failed", path=str(feedback_path), error=str(exc))
        _feedback_texts_cache = fallback_texts
        return _feedback_texts_cache

    status_text = raw_data.get("STATUS_TEXT") if isinstance(raw_data, dict) else None
    if not isinstance(status_text, dict):
        _feedback_texts_cache = fallback_texts
        return _feedback_texts_cache

    texts = [
        value.strip()
        for key, value in status_text.items()
        if key != "failed" and isinstance(value, str) and value.strip()
    ]
    _feedback_texts_cache = texts or fallback_texts
    return _feedback_texts_cache


def _get_initial_feedback_text() -> str:
    feedback_texts = _load_feedback_texts()
    return feedback_texts[0] if feedback_texts else "🤔 正在理解问题..."


def _get_failed_feedback_text() -> str:
    fallback_text = "⚠️ 处理失败，请稍后重试"
    feedback_path = Path("data") / "UI反馈.json"
    try:
        raw_data = json.loads(feedback_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback_text

    status_text = raw_data.get("STATUS_TEXT") if isinstance(raw_data, dict) else None
    failed_text = status_text.get("failed") if isinstance(status_text, dict) else None
    return failed_text.strip() if isinstance(failed_text, str) and failed_text.strip() else fallback_text


def _get_unsupported_message_text() -> str:
    return "暂不支持这种消息内容，请发送文字、Markdown、列表或链接。"


async def _send_unsupported_feishu_message_notice(
    *,
    chat_id: str | None,
    message_id: str | None,
) -> None:
    unsupported_text = _get_unsupported_message_text()
    if message_id:
        sent = await _try_reply_feishu_markdown(message_id, unsupported_text)
        if sent:
            return
    if chat_id:
        await _try_send_feishu_markdown_message(
            chat_id=chat_id,
            reply_to_message_id=None,
            markdown_text=unsupported_text,
        )


async def _try_send_feishu_markdown_message(
    *,
    chat_id: str | None,
    reply_to_message_id: str | None,
    markdown_text: str,
    log_content: bool = True,
    stop_cancel_id: str | None = None,
    canceled_text: str | None = None,
    answer_feedback_id: str | None = None,
    answer_feedback_selected: str | None = None,
) -> str | None:
    if chat_id:
        fields: dict[str, Any] = {"chat_id": chat_id, "markdown_len": len(markdown_text)}
        if log_content:
            fields["markdown_preview"] = _preview(markdown_text)
        _debug("markdown send attempt", **fields)
        try:
            return await _send_feishu_markdown_message_to_chat(
                chat_id,
                markdown_text,
                log_content=log_content,
                stop_cancel_id=stop_cancel_id,
                canceled_text=canceled_text,
                answer_feedback_id=answer_feedback_id,
                answer_feedback_selected=answer_feedback_selected,
            )
        except HTTPException as exc:
            if (stop_cancel_id or answer_feedback_id) and _is_feishu_card_content_error(exc):
                _warn(
                    "interactive card controls rejected; retrying without controls",
                    chat_id=chat_id,
                    error=_detail_to_text(exc.detail),
                )
                try:
                    return await _send_feishu_markdown_message_to_chat(
                        chat_id,
                        markdown_text,
                        log_content=log_content,
                        canceled_text=canceled_text,
                    )
                except HTTPException as fallback_exc:
                    exc = fallback_exc
            detail = _detail_to_text(exc.detail)
            logger.exception("Failed to send Feishu markdown card to chat %s: %s", chat_id, detail)
            _debug("markdown send failed", chat_id=chat_id, error=detail)

    if reply_to_message_id:
        return await _try_send_feishu_markdown_reply(
            reply_to_message_id,
            markdown_text,
            log_content=log_content,
            stop_cancel_id=stop_cancel_id,
            canceled_text=canceled_text,
            answer_feedback_id=answer_feedback_id,
            answer_feedback_selected=answer_feedback_selected,
        )

    return None


async def _try_send_feishu_markdown_reply(
    message_id: str,
    markdown_text: str,
    *,
    log_content: bool = True,
    stop_cancel_id: str | None = None,
    canceled_text: str | None = None,
    answer_feedback_id: str | None = None,
    answer_feedback_selected: str | None = None,
) -> str | None:
    fields: dict[str, Any] = {"message_id": message_id, "markdown_len": len(markdown_text)}
    if log_content:
        fields["markdown_preview"] = _preview(markdown_text)
    _debug("markdown reply attempt", **fields)
    try:
        return await _send_feishu_markdown_reply(
            message_id,
            markdown_text,
            log_content=log_content,
            stop_cancel_id=stop_cancel_id,
            canceled_text=canceled_text,
            answer_feedback_id=answer_feedback_id,
            answer_feedback_selected=answer_feedback_selected,
        )
    except HTTPException as exc:
        if (stop_cancel_id or answer_feedback_id) and _is_feishu_card_content_error(exc):
            _warn(
                "interactive reply controls rejected; retrying without controls",
                message_id=message_id,
                error=_detail_to_text(exc.detail),
            )
            try:
                return await _send_feishu_markdown_reply(
                    message_id,
                    markdown_text,
                    log_content=log_content,
                    canceled_text=canceled_text,
                )
            except HTTPException as fallback_exc:
                exc = fallback_exc
        detail = _detail_to_text(exc.detail)
        logger.exception("Failed to send Feishu markdown card %s: %s", message_id, detail)
        _debug("markdown reply failed", message_id=message_id, error=detail)
        return None


async def _try_reply_feishu_markdown(message_id: str, markdown_text: str) -> bool:
    return await _try_send_feishu_markdown_reply(message_id, markdown_text) is not None


async def _try_update_feishu_markdown_message(
    message_id: str,
    markdown_text: str,
    *,
    log_content: bool = True,
    stop_cancel_id: str | None = None,
    canceled_text: str | None = None,
    answer_feedback_id: str | None = None,
    answer_feedback_selected: str | None = None,
) -> bool:
    fields: dict[str, Any] = {"message_id": message_id, "markdown_len": len(markdown_text)}
    if log_content:
        fields["markdown_preview"] = _preview(markdown_text)
    _debug("markdown update attempt", **fields)
    try:
        return await _update_feishu_markdown_message(
            message_id,
            markdown_text,
            log_content=log_content,
            stop_cancel_id=stop_cancel_id,
            canceled_text=canceled_text,
            answer_feedback_id=answer_feedback_id,
            answer_feedback_selected=answer_feedback_selected,
        )
    except HTTPException as exc:
        if (stop_cancel_id or answer_feedback_id) and _is_feishu_card_content_error(exc):
            _warn(
                "interactive update controls rejected; retrying without controls",
                message_id=message_id,
                error=_detail_to_text(exc.detail),
            )
            try:
                return await _update_feishu_markdown_message(
                    message_id,
                    markdown_text,
                    log_content=log_content,
                    canceled_text=canceled_text,
                )
            except HTTPException as fallback_exc:
                exc = fallback_exc
        detail = _detail_to_text(exc.detail)
        logger.exception("Failed to update Feishu markdown card %s: %s", message_id, detail)
        _debug("markdown update failed", message_id=message_id, error=detail)
        return False


def _register_feishu_answer_cancel_state(
    *,
    question: str,
    chat_id: str | None,
    message_id: str | None,
) -> str:
    _cleanup_feishu_answer_cancel_states()
    cancel_id = uuid.uuid4().hex
    _feishu_answer_cancel_states[cancel_id] = _FeishuAnswerCancelState(
        cancel_id=cancel_id,
        started_at=time.monotonic(),
        incoming_message_id=message_id,
        chat_id=chat_id,
        question=question,
        answer_task=asyncio.current_task(),
    )
    return cancel_id


async def _register_feishu_answer_feedback_state(answer: str) -> str:
    await _cleanup_feishu_answer_feedback_states()
    feedback_id = uuid.uuid4().hex
    _feishu_answer_feedback_states[feedback_id] = _FeishuAnswerFeedbackState(
        feedback_id=feedback_id,
        answer=answer,
        selected=None,
        created_at=time.monotonic(),
    )
    try:
        await asyncio.to_thread(
            insert_feishu_answer_feedback_state,
            feedback_id=feedback_id,
            answer=encrypt_chat_text(answer),
        )
    except Exception as exc:  # noqa: BLE001 - in-memory state still supports same-worker clicks.
        _warn("answer feedback state persist failed", feedback_id=feedback_id, error=str(exc))
    return feedback_id


async def _cleanup_feishu_answer_feedback_states() -> None:
    now = time.monotonic()
    ttl = _answer_feedback_window_seconds()
    expired_ids = [
        feedback_id
        for feedback_id, state in _feishu_answer_feedback_states.items()
        if now - state.created_at > ttl
    ]
    for feedback_id in expired_ids:
        _feishu_answer_feedback_states.pop(feedback_id, None)
    try:
        await asyncio.to_thread(
            delete_expired_feishu_answer_feedback_states,
            older_than_seconds=ttl,
        )
    except Exception as exc:  # noqa: BLE001 - cleanup must not block feedback.
        _warn("answer feedback state cleanup failed", error=str(exc))


def _answer_feedback_window_seconds() -> float:
    return max(float(settings.feishu_feedback_window_seconds), 300.0)


def _bind_feishu_answer_feedback(
    cancel_id: str | None,
    feedback_handle: _FeishuFeedbackHandle | None,
) -> None:
    if not cancel_id or not feedback_handle:
        return
    cancel_state = _feishu_answer_cancel_states.get(cancel_id)
    if cancel_state:
        cancel_state.feedback_handle = feedback_handle


def _discard_feishu_answer_cancel_state(cancel_id: str | None) -> None:
    if cancel_id:
        cancel_state = _feishu_answer_cancel_states.get(cancel_id)
        if cancel_state and cancel_state.canceled_at is not None:
            return
        _feishu_answer_cancel_states.pop(cancel_id, None)


def _cleanup_feishu_answer_cancel_states() -> None:
    now = time.monotonic()
    expired_ids = [
        cancel_id
        for cancel_id, state in _feishu_answer_cancel_states.items()
        if now - state.started_at > 7200
    ]
    for cancel_id in expired_ids:
        _feishu_answer_cancel_states.pop(cancel_id, None)


def _is_feishu_answer_cancelled(cancel_id: str | None) -> bool:
    if not cancel_id:
        return False
    cancel_state = _feishu_answer_cancel_states.get(cancel_id)
    return bool(cancel_state and cancel_state.canceled_at is not None)


def _is_feishu_card_action_event(event_type: Any, payload: dict[str, Any]) -> bool:
    if event_type == "card.action.trigger":
        return True
    if payload.get("type") == "card.action.trigger":
        return True
    action = _extract_feishu_card_action_value(payload)
    return isinstance(action, dict) and action.get("action") in {
        "stop_feishu_answer",
        "answer_feedback",
    }


async def _handle_feishu_card_action(payload: dict[str, Any]) -> dict[str, Any]:
    value = _extract_feishu_card_action_value(payload)
    if not isinstance(value, dict):
        return {"ok": True, "ignored": True, "reason": "unsupported_card_action"}
    if value.get("action") == "answer_feedback":
        return await _handle_feishu_answer_feedback_action(value)
    if value.get("action") != "stop_feishu_answer":
        return {"ok": True, "ignored": True, "reason": "unsupported_card_action"}

    cancel_id = value.get("cancel_id")
    if not isinstance(cancel_id, str) or not cancel_id:
        return {"ok": False, "ignored": True, "reason": "missing_cancel_id"}

    cancelled = await _cancel_feishu_answer(cancel_id)
    result: dict[str, Any] = {
        "ok": True,
        "cancelled": cancelled,
        "toast": {
            "type": "info",
            "content": "已停止回答" if cancelled else "回答已结束或无法停止",
        },
    }
    callback_card = await _build_feishu_cancel_callback_card(cancel_id)
    if callback_card:
        result["card"] = {
            "type": "raw",
            "data": callback_card,
        }
    return result


def _extract_feishu_card_action_value(payload: dict[str, Any]) -> dict[str, Any] | None:
    event = payload.get("event")
    if not isinstance(event, dict):
        event = {}

    candidates: list[Any] = [
        event.get("action"),
        payload.get("action"),
        event.get("action_info"),
        payload.get("action_info"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        value = candidate.get("value")
        if isinstance(value, dict):
            return value
        if candidate.get("action") in {"stop_feishu_answer", "answer_feedback"}:
            return candidate
    return None


async def _handle_feishu_answer_feedback_action(value: dict[str, Any]) -> dict[str, Any]:
    feedback_id = value.get("feedback_id")
    choice = value.get("choice")
    if not isinstance(feedback_id, str) or not feedback_id:
        return {"ok": False, "ignored": True, "reason": "missing_feedback_id"}
    if choice not in {"helpful", "issue"}:
        return {"ok": False, "ignored": True, "reason": "invalid_feedback_choice"}

    await _cleanup_feishu_answer_feedback_states()
    feedback_state = await _get_feishu_answer_feedback_state(feedback_id)
    if not feedback_state:
        return {
            "ok": True,
            "ignored": True,
            "reason": "feedback_state_expired",
            "toast": {
                "type": "info",
                "content": "反馈入口已过期，可以通过卡片里的链接继续提交问题。",
            },
        }

    if feedback_state.selected is None:
        feedback_state = await _select_feishu_answer_feedback(feedback_state, choice)
        if feedback_state.selected == choice:
            toast_content = (
                "已收到，你的反馈会帮助我们继续优化知识库。"
                if choice == "helpful"
                else "已为你打开反馈表，请补充一下问题细节。"
            )
        else:
            toast_content = "这条回答已经记录过反馈了。"
    else:
        toast_content = "这条回答已经记录过反馈了。"

    result: dict[str, Any] = {
        "ok": True,
        "selected": feedback_state.selected,
        "toast": {
            "type": "success" if feedback_state.selected == choice else "info",
            "content": toast_content,
        },
    }
    token = await _get_tenant_access_token() if settings.feishu_app_id and settings.feishu_app_secret else ""
    content = await _build_feishu_card_content(
        feedback_state.answer,
        token,
        answer_feedback_id=feedback_state.feedback_id,
        answer_feedback_selected=feedback_state.selected,
    )
    result["card"] = {
        "type": "raw",
        "data": json.loads(content),
    }
    return result


async def _get_feishu_answer_feedback_state(
    feedback_id: str,
) -> _FeishuAnswerFeedbackState | None:
    feedback_state = _feishu_answer_feedback_states.get(feedback_id)
    if feedback_state:
        return feedback_state

    try:
        row = await asyncio.to_thread(get_feishu_answer_feedback_state, feedback_id)
    except Exception as exc:  # noqa: BLE001 - fall through to expired UX.
        _warn("answer feedback state load failed", feedback_id=feedback_id, error=str(exc))
        return None

    if not row:
        return None

    created_at = row.get("create_time")
    if isinstance(created_at, datetime):
        now = datetime.now(created_at.tzinfo or timezone.utc)
        if (now - created_at).total_seconds() > _answer_feedback_window_seconds():
            return None

    encrypted_answer = str(row.get("answer") or "")
    try:
        answer = decrypt_chat_text(encrypted_answer)
    except Exception as exc:  # noqa: BLE001 - corrupted state should not break card callbacks.
        _warn("answer feedback state decrypt failed", feedback_id=feedback_id, error=str(exc))
        return None

    selected = row.get("selected")
    selected_text = selected if selected in {"helpful", "issue"} else None
    feedback_state = _FeishuAnswerFeedbackState(
        feedback_id=feedback_id,
        answer=answer,
        selected=selected_text,
        created_at=time.monotonic(),
    )
    _feishu_answer_feedback_states[feedback_id] = feedback_state
    return feedback_state


async def _select_feishu_answer_feedback(
    feedback_state: _FeishuAnswerFeedbackState,
    choice: str,
) -> _FeishuAnswerFeedbackState:
    try:
        row = await asyncio.to_thread(
            update_feishu_answer_feedback_selection,
            feedback_id=feedback_state.feedback_id,
            selected=choice,
        )
    except Exception as exc:  # noqa: BLE001 - in-memory state still gives immediate feedback.
        _warn(
            "answer feedback state selection persist failed",
            feedback_id=feedback_state.feedback_id,
            error=str(exc),
        )
        feedback_state.selected = choice
        return feedback_state

    selected = row.get("selected") if row else None
    feedback_state.selected = selected if selected in {"helpful", "issue"} else choice
    return feedback_state


async def _cancel_feishu_answer(cancel_id: str) -> bool:
    cancel_state = _feishu_answer_cancel_states.get(cancel_id)
    if not cancel_state:
        _debug("answer cancel ignored", reason="missing_cancel_state", cancel_id=cancel_id)
        return False

    if cancel_state.canceled_at is None:
        cancel_state.canceled_at = time.monotonic()

    feedback_handle = cancel_state.feedback_handle
    if feedback_handle:
        feedback_handle.stop_event.set()
        canceled_text = _format_feishu_cancel_elapsed(
            started_at=cancel_state.started_at,
            canceled_at=cancel_state.canceled_at,
        )
        state = feedback_handle.state
        state.canceled_text = canceled_text
        await _cancel_feishu_feedback_updates(feedback_handle)
        await _rewrite_cancelled_feishu_feedback_card(state)

    answer_task = cancel_state.answer_task
    current_task = asyncio.current_task()
    if answer_task and answer_task is not current_task and not answer_task.done():
        answer_task.cancel()

    _debug(
        "answer cancel requested",
        cancel_id=cancel_id,
        incoming_message_id=cancel_state.incoming_message_id,
        chat_id=cancel_state.chat_id,
        has_feedback=bool(feedback_handle),
    )
    return True


async def _build_feishu_cancel_callback_card(cancel_id: str) -> dict[str, Any] | None:
    cancel_state = _feishu_answer_cancel_states.get(cancel_id)
    feedback_handle = cancel_state.feedback_handle if cancel_state else None
    if not feedback_handle:
        return None

    state = feedback_handle.state
    if not state.canceled_text:
        if not cancel_state or cancel_state.canceled_at is None:
            return None
        state.canceled_text = _format_feishu_cancel_elapsed(
            started_at=cancel_state.started_at,
            canceled_at=cancel_state.canceled_at,
        )

    content = await _build_feishu_card_content(
        _compose_cancelled_feedback_card_text(state),
        "",
        canceled_text=state.canceled_text,
    )
    return json.loads(content)


async def _cancel_feishu_feedback_updates(handle: _FeishuFeedbackHandle) -> None:
    handle.stop_event.set()
    for task in (handle.task, handle.comfort_task):
        if not task.done():
            task.cancel()
    await asyncio.gather(handle.task, handle.comfort_task, return_exceptions=True)


async def _rewrite_cancelled_feishu_feedback_card(state: _FeishuFeedbackState) -> bool:
    async with state.update_lock:
        return await _try_update_feishu_markdown_message(
            state.message_id,
            _compose_cancelled_feedback_card_text(state),
            log_content=False,
            canceled_text=state.canceled_text,
        )


def _format_feishu_cancel_elapsed(*, started_at: float, canceled_at: float) -> str:
    elapsed_seconds = max(0, int(round(canceled_at - started_at)))
    minutes, seconds = divmod(elapsed_seconds, 60)
    if minutes <= 0:
        return f"您在{seconds}秒后取消回答"
    return f"您在{minutes}分{seconds}秒后取消回答"


def _verify_feishu_token(payload: dict[str, Any]) -> None:
    expected_token = settings.feishu_verification_token
    if not expected_token:
        _debug("verification token skipped", reason="not_configured")
        return

    actual_token = payload.get("token")
    if not actual_token:
        header = payload.get("header")
        if isinstance(header, dict):
            actual_token = header.get("token")

    if actual_token != expected_token:
        _debug("verification token failed")
        raise HTTPException(status_code=401, detail="Invalid Feishu verification token")

    _debug("verification token passed")


def _extract_message_text(message: dict[str, Any]) -> str:
    message_type = message.get("message_type")
    content = _parse_feishu_message_content(message.get("content"))

    if message_type == "text":
        if isinstance(content, dict):
            text = content.get("text")
            return text.strip() if isinstance(text, str) else ""
        return content.strip() if isinstance(content, str) else ""

    if message_type == "post":
        return _extract_feishu_post_text(content)

    return ""


def _parse_feishu_message_content(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content.strip()
    return content


def _extract_feishu_post_text(content: Any) -> str:
    if not isinstance(content, (dict, list)):
        return content.strip() if isinstance(content, str) else ""

    if isinstance(content, dict):
        for key in ("content", "elements", "children", "items"):
            if key in content:
                lines = _extract_feishu_post_lines(content[key])
                if lines:
                    return "\n".join(lines).strip()

    return "\n".join(_extract_feishu_post_lines(content)).strip()


def _extract_feishu_post_lines(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []

    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            if isinstance(item, list):
                lines.extend(_split_feishu_post_line(_extract_feishu_inline_text(item)))
            else:
                lines.extend(_extract_feishu_post_lines(item))
        return lines

    if not isinstance(value, dict):
        return []

    tag = str(value.get("tag") or value.get("type") or "").lower()
    if tag in {"li", "list_item"}:
        text = _extract_feishu_inline_text(
            value.get("content")
            or value.get("elements")
            or value.get("children")
            or value.get("items")
        )
        return [f"- {text.strip()}"] if text.strip() else []
    if tag in {"a", "link"}:
        link_markdown = _extract_feishu_link_markdown(value)
        return [link_markdown] if link_markdown else []

    direct_text = _extract_feishu_direct_text(value)
    if direct_text:
        return _split_feishu_post_line(direct_text)

    lines: list[str] = []
    for key in ("content", "elements", "children", "items"):
        if key in value:
            lines.extend(_extract_feishu_post_lines(value[key]))
    return lines


def _extract_feishu_inline_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "".join(_extract_feishu_inline_text(item) for item in value)

    if not isinstance(value, dict):
        return ""

    tag = str(value.get("tag") or value.get("type") or "").lower()
    if tag in {"li", "list_item"}:
        item_text = _extract_feishu_inline_text(
            value.get("content")
            or value.get("elements")
            or value.get("children")
            or value.get("items")
        ).strip()
        return f"- {item_text}" if item_text else ""
    if tag in {"a", "link"}:
        return _extract_feishu_link_markdown(value)
    if tag in {"at", "mention"}:
        name = value.get("user_name") or value.get("name") or value.get("text")
        return f"@{name.strip()}" if isinstance(name, str) and name.strip() else ""

    direct_text = _extract_feishu_direct_text(value)
    if direct_text:
        return direct_text

    parts: list[str] = []
    for key in ("content", "elements", "children", "items"):
        if key in value:
            parts.append(_extract_feishu_inline_text(value[key]))
    return "".join(parts)


def _extract_feishu_direct_text(value: dict[str, Any]) -> str:
    for key in ("text", "un_escape", "user_name", "name"):
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            return text
    return ""


def _extract_feishu_link_markdown(value: dict[str, Any]) -> str:
    href = value.get("href") or value.get("url")
    text = value.get("text") or value.get("name") or href
    if not isinstance(href, str) or not href.strip():
        return text.strip() if isinstance(text, str) else ""
    label = text.strip() if isinstance(text, str) and text.strip() else href.strip()
    return f"[{label}]({href.strip()})"


def _split_feishu_post_line(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _extract_sender_id(sender: dict[str, Any]) -> str | None:
    sender_id = sender.get("sender_id")
    if not isinstance(sender_id, dict):
        return None
    for key in ("union_id", "open_id", "user_id"):
        value = sender_id.get(key)
        if value:
            return str(value)
    return None


def _extract_sender_name(sender: dict[str, Any]) -> str | None:
    for key in ("user_name", "sender_name", "name", "display_name", "nickname"):
        value = sender.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    sender_id = sender.get("sender_id")
    if isinstance(sender_id, dict):
        for key in ("user_name", "name", "display_name", "nickname"):
            value = sender_id.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _build_dedupe_key(message_id: Any, event_id: Any) -> str | None:
    if message_id:
        return f"message:{message_id}"
    if event_id:
        return f"event:{event_id}"
    return None


def _is_duplicate_message(dedupe_key: str | None) -> bool:
    if not dedupe_key:
        _debug("dedupe skipped", reason="missing_message_id_and_event_id")
        return False

    now = time.time()
    expired_keys = [
        cached_key
        for cached_key, expires_at in _seen_message_keys.items()
        if expires_at <= now
    ]
    for cached_key in expired_keys:
        _seen_message_keys.pop(cached_key, None)

    if dedupe_key in _seen_message_keys:
        return True

    _seen_message_keys[dedupe_key] = now + _seen_message_ttl_seconds
    _debug("message cached for dedupe", dedupe_key=dedupe_key)
    return False


async def _send_feishu_markdown_message_to_chat(
    chat_id: str,
    markdown_text: str,
    *,
    log_content: bool = True,
    stop_cancel_id: str | None = None,
    canceled_text: str | None = None,
    answer_feedback_id: str | None = None,
    answer_feedback_selected: str | None = None,
) -> str | None:
    if not settings.feishu_app_id or not settings.feishu_app_secret:
        _debug(
            "markdown send skipped",
            reason="missing_app_credentials",
            has_app_id=bool(settings.feishu_app_id),
            has_app_secret=bool(settings.feishu_app_secret),
        )
        return None

    token = await _get_tenant_access_token()
    timeout = httpx.Timeout(settings.feishu_timeout)
    url = f"{settings.feishu_base_url}/open-apis/im/v1/messages"
    content = await _build_feishu_card_content(
        markdown_text,
        token,
        stop_cancel_id=stop_cancel_id,
        canceled_text=canceled_text,
        answer_feedback_id=answer_feedback_id,
        answer_feedback_selected=answer_feedback_selected,
    )
    fields: dict[str, Any] = {"chat_id": chat_id, "url": url}
    if log_content:
        fields["content_preview"] = _preview(content)
    _debug("markdown send api request", **fields)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                params={"receive_id_type": "chat_id"},
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "receive_id": chat_id,
                    "msg_type": "interactive",
                    "content": content,
                },
            )
            _debug(
                "markdown send api response",
                chat_id=chat_id,
                http_status=response.status_code,
                body_preview=_preview(response.text),
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Feishu markdown send API timed out") from exc
    except httpx.HTTPStatusError as exc:
        detail = _response_text(exc.response)
        raise HTTPException(
            status_code=502,
            detail=f"Feishu markdown send API returned {exc.response.status_code}: {detail}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to call Feishu markdown send API: {exc}",
        ) from exc

    result = response.json()
    if result.get("code") != 0:
        raise HTTPException(
            status_code=502,
            detail=f"Feishu markdown send API error: {result}",
        )

    sent_message_id = _extract_feishu_message_id(result)
    _debug(
        "markdown send api success",
        chat_id=chat_id,
        sent_message_id=sent_message_id,
    )
    return sent_message_id


async def _send_feishu_markdown_reply(
    message_id: str,
    markdown_text: str,
    *,
    log_content: bool = True,
    stop_cancel_id: str | None = None,
    canceled_text: str | None = None,
    answer_feedback_id: str | None = None,
    answer_feedback_selected: str | None = None,
) -> str | None:
    if not settings.feishu_app_id or not settings.feishu_app_secret:
        _debug(
            "markdown reply skipped",
            reason="missing_app_credentials",
            has_app_id=bool(settings.feishu_app_id),
            has_app_secret=bool(settings.feishu_app_secret),
        )
        return None

    token = await _get_tenant_access_token()
    timeout = httpx.Timeout(settings.feishu_timeout)
    url = f"{settings.feishu_base_url}/open-apis/im/v1/messages/{message_id}/reply"
    content = await _build_feishu_card_content(
        markdown_text,
        token,
        stop_cancel_id=stop_cancel_id,
        canceled_text=canceled_text,
        answer_feedback_id=answer_feedback_id,
        answer_feedback_selected=answer_feedback_selected,
    )
    fields: dict[str, Any] = {"message_id": message_id, "url": url}
    if log_content:
        fields["content_preview"] = _preview(content)
    _debug("markdown reply api request", **fields)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "msg_type": "interactive",
                    "content": content,
                },
            )
            _debug(
                "markdown reply api response",
                message_id=message_id,
                http_status=response.status_code,
                body_preview=_preview(response.text),
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Feishu markdown reply API timed out") from exc
    except httpx.HTTPStatusError as exc:
        detail = _response_text(exc.response)
        raise HTTPException(
            status_code=502,
            detail=f"Feishu markdown reply API returned {exc.response.status_code}: {detail}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to call Feishu markdown reply API: {exc}",
        ) from exc

    result = response.json()
    if result.get("code") != 0:
        raise HTTPException(
            status_code=502,
            detail=f"Feishu markdown reply API error: {result}",
        )

    reply_message_id = _extract_feishu_message_id(result)
    _debug(
        "markdown reply api success",
        message_id=message_id,
        reply_message_id=reply_message_id,
    )
    return reply_message_id


async def _update_feishu_markdown_message(
    message_id: str,
    markdown_text: str,
    *,
    log_content: bool = True,
    stop_cancel_id: str | None = None,
    canceled_text: str | None = None,
    answer_feedback_id: str | None = None,
    answer_feedback_selected: str | None = None,
) -> bool:
    if not settings.feishu_app_id or not settings.feishu_app_secret:
        _debug(
            "markdown update skipped",
            reason="missing_app_credentials",
            has_app_id=bool(settings.feishu_app_id),
            has_app_secret=bool(settings.feishu_app_secret),
        )
        return False

    token = await _get_tenant_access_token()
    timeout = httpx.Timeout(settings.feishu_timeout)
    url = f"{settings.feishu_base_url}/open-apis/im/v1/messages/{message_id}"
    content = await _build_feishu_card_content(
        markdown_text,
        token,
        stop_cancel_id=stop_cancel_id,
        canceled_text=canceled_text,
        answer_feedback_id=answer_feedback_id,
        answer_feedback_selected=answer_feedback_selected,
    )
    fields: dict[str, Any] = {"message_id": message_id, "url": url}
    if log_content:
        fields["content_preview"] = _preview(content)
    _debug("markdown update api request", **fields)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.patch(
                url,
                headers={"Authorization": f"Bearer {token}"},
                json={"content": content},
            )
            _debug(
                "markdown update api response",
                message_id=message_id,
                http_status=response.status_code,
                body_preview=_preview(response.text),
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Feishu markdown update API timed out") from exc
    except httpx.HTTPStatusError as exc:
        detail = _response_text(exc.response)
        raise HTTPException(
            status_code=502,
            detail=f"Feishu markdown update API returned {exc.response.status_code}: {detail}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to call Feishu markdown update API: {exc}",
        ) from exc

    result = response.json()
    if result.get("code") != 0:
        raise HTTPException(
            status_code=502,
            detail=f"Feishu markdown update API error: {result}",
        )

    _debug("markdown update api success", message_id=message_id)
    return True


def _extract_feishu_message_id(result: dict[str, Any]) -> str | None:
    data = result.get("data")
    if not isinstance(data, dict):
        return None

    for key in ("message_id", "messageId"):
        value = data.get(key)
        if value:
            return str(value)

    message = data.get("message")
    if isinstance(message, dict):
        value = message.get("message_id") or message.get("messageId")
        if value:
            return str(value)

    return None


async def _build_feishu_card_content(
    markdown_text: str,
    token: str,
    *,
    stop_cancel_id: str | None = None,
    canceled_text: str | None = None,
    answer_feedback_id: str | None = None,
    answer_feedback_selected: str | None = None,
) -> str:
    markdown_text, reference_sources = _rewrite_reference_links(markdown_text)
    markdown_text = _render_latex_for_feishu(markdown_text)
    elements = await _build_feishu_card_elements(markdown_text, token)
    if reference_sources:
        elements.append(_build_reference_sources_panel(reference_sources))
    if canceled_text:
        elements.extend(_build_cancelled_feedback_footer(canceled_text))
    elif stop_cancel_id:
        elements.append(_build_stop_answer_button(stop_cancel_id))
    if answer_feedback_id:
        elements.extend(
            _build_answer_feedback_footer(
                feedback_id=answer_feedback_id,
                selected=answer_feedback_selected,
            )
        )
    if not elements:
        elements = [_build_markdown_element(" ")]

    return json.dumps(
        {
            "schema": "2.0",
            "config": {
                "update_multi": True,
            },
            "body": {
                "direction": "vertical",
                "elements": elements,
            },
        },
        ensure_ascii=False,
    )


def _build_stop_answer_button(cancel_id: str) -> dict[str, Any]:
    return {
        "tag": "button",
        "text": {
            "tag": "plain_text",
            "content": "⏹ 停止回答",
        },
        "type": "danger",
        "width": "default",
        "behaviors": [
            {
                "type": "callback",
                "value": {
                    "action": "stop_feishu_answer",
                    "cancel_id": cancel_id,
                },
            }
        ],
    }


def _build_answer_feedback_footer(
    *,
    feedback_id: str,
    selected: str | None = None,
) -> list[dict[str, Any]]:
    selected = selected if selected in {"helpful", "issue"} else None
    feedback_elements = [
        _build_markdown_element(
            "<font color='grey'>这条回答对你有帮助吗？你的反馈会帮助知识库持续变好。</font>"
        ),
        {
            "tag": "column_set",
            "flex_mode": "bisect",
            "background_style": "default",
            "columns": [
                {
                    "tag": "column",
                    "width": "weighted",
                    "weight": 1,
                    "vertical_align": "top",
                    "elements": [
                        _build_answer_feedback_button(
                            feedback_id=feedback_id,
                            choice="helpful",
                            selected=selected,
                        )
                    ],
                },
                {
                    "tag": "column",
                    "width": "weighted",
                    "weight": 1,
                    "vertical_align": "top",
                    "elements": [
                        _build_answer_feedback_button(
                            feedback_id=feedback_id,
                            choice="issue",
                            selected=selected,
                        )
                    ],
                },
            ],
        },
        _build_markdown_element(
            f"<font color='grey'>有问题也可以直接[去反馈]({_format_markdown_link_url(_get_answer_feedback_form_url())})。</font>"
        ),
    ]
    return [
        {
            "tag": "collapsible_panel",
            "expanded": False,
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": "反馈",
                }
            },
            "elements": feedback_elements,
        }
    ]


def _build_answer_feedback_button(
    *,
    feedback_id: str,
    choice: str,
    selected: str | None,
) -> dict[str, Any]:
    is_selected = selected == choice
    is_locked = selected is not None
    if choice == "helpful":
        content = "👍 已标记有帮助" if is_selected else "👍 有帮助"
        button_type = "primary" if is_selected else "default"
        behaviors = [
            {
                "type": "callback",
                "value": {
                    "action": "answer_feedback",
                    "feedback_id": feedback_id,
                    "choice": "helpful",
                },
            }
        ]
    else:
        content = "👎 已选择去反馈" if is_selected else "👎 有问题，去反馈"
        button_type = "danger" if is_selected else "default"
        behaviors = [
            {
                "type": "callback",
                "value": {
                    "action": "answer_feedback",
                    "feedback_id": feedback_id,
                    "choice": "issue",
                },
            },
            {
                "type": "open_url",
                "default_url": _get_answer_feedback_form_url(),
                "pc_url": _get_answer_feedback_form_url(),
                "ios_url": _get_answer_feedback_form_url(),
                "android_url": _get_answer_feedback_form_url(),
            },
        ]

    button: dict[str, Any] = {
        "tag": "button",
        "text": {
            "tag": "plain_text",
            "content": content,
        },
        "type": button_type,
        "width": "fill",
        "behaviors": behaviors,
    }
    if is_locked and not is_selected:
        button["disabled"] = True
    return button


def _get_answer_feedback_form_url() -> str:
    return settings.feishu_feedback_form_url or (
        "https://tmqhw1h9zt.feishu.cn/wiki/LbjCwUPA6iUbF5k2SFbcowT8nne"
    )


def _build_cancelled_feedback_footer(canceled_text: str) -> list[dict[str, Any]]:
    return [
        {"tag": "hr"},
        _build_markdown_element(f"<font color='grey'>{canceled_text}</font>"),
    ]


async def _build_feishu_card_elements(markdown_text: str, token: str) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    markdown_buffer: list[str] = []
    image_buffer: list[_PendingImage] = []
    cursor = 0

    for match in _pseudo_tag_pattern.finditer(markdown_text):
        before_tag = markdown_text[cursor : match.start()]
        tag = match.group(1).lower()
        value = match.group(2).strip()

        if tag == "link":
            _flush_image_buffer(elements, image_buffer)
            markdown_buffer.append(before_tag)
            markdown_buffer.append(_link_tag_to_markdown(value))
        elif tag == "img":
            markdown_buffer.append(before_tag)
            if "".join(markdown_buffer).strip():
                _flush_image_buffer(elements, image_buffer)
                _flush_markdown_buffer(elements, markdown_buffer)
            else:
                markdown_buffer.clear()
            image_key = await _upload_local_image(value, token)
            if image_key:
                alt_text = Path(value.replace("\\", "/")).name or "image"
                image_buffer.append(
                    _PendingImage(
                        image_key=image_key,
                        alt_text=alt_text,
                        size=_get_local_image_size(value),
                    )
                )

        cursor = match.end()

    markdown_buffer.append(markdown_text[cursor:])
    if image_buffer:
        if "".join(markdown_buffer).strip():
            _flush_image_buffer(elements, image_buffer)
            _flush_markdown_buffer(elements, markdown_buffer)
        else:
            markdown_buffer.clear()
            _flush_image_buffer(elements, image_buffer)
    else:
        _flush_markdown_buffer(elements, markdown_buffer)
    return elements


def _build_image_element(image_key: str, alt_text: str) -> dict[str, Any]:
    return {
        "tag": "img",
        "img_key": image_key,
        "alt": {
            "tag": "plain_text",
            "content": alt_text,
        },
    }


def _flush_image_buffer(elements: list[dict[str, Any]], images: list[_PendingImage]) -> None:
    index = 0
    while index < len(images):
        current = images[index]
        next_image = images[index + 1] if index + 1 < len(images) else None
        if next_image and _image_sizes_are_similar(current.size, next_image.size):
            elements.append(_build_two_column_image_element(current, next_image))
            index += 2
            continue
        elements.append(_build_half_width_image_element(current))
        index += 1
    images.clear()


def _build_half_width_image_element(image: _PendingImage) -> dict[str, Any]:
    return _build_image_column_set([image, None])


def _build_two_column_image_element(left: _PendingImage, right: _PendingImage) -> dict[str, Any]:
    return _build_image_column_set([left, right])


def _build_image_column_set(images: list[_PendingImage | None]) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    for image in images:
        columns.append(
            {
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "vertical_align": "top",
                "elements": [
                    _build_image_element(image.image_key, image.alt_text)
                ]
                if image
                else [],
            }
        )
    return {
        "tag": "column_set",
        "flex_mode": "bisect",
        "background_style": "default",
        "columns": columns,
    }


def _image_sizes_are_similar(
    left: tuple[int, int] | None,
    right: tuple[int, int] | None,
) -> bool:
    if not left or not right:
        return False
    left_width, left_height = left
    right_width, right_height = right
    if min(left_width, left_height, right_width, right_height) <= 0:
        return False

    left_area = left_width * left_height
    right_area = right_width * right_height
    area_ratio = min(left_area, right_area) / max(left_area, right_area)
    left_aspect = left_width / left_height
    right_aspect = right_width / right_height
    aspect_ratio = min(left_aspect, right_aspect) / max(left_aspect, right_aspect)
    return area_ratio >= 0.55 and aspect_ratio >= 0.65


def _get_local_image_size(raw_path: str) -> tuple[int, int] | None:
    if Image is None:
        return None
    image_path = _resolve_local_image_path(raw_path)
    if image_path is None:
        return None
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        return None
    return int(width), int(height)


def _flush_markdown_buffer(elements: list[dict[str, Any]], buffer: list[str]) -> None:
    markdown = "".join(buffer).strip()
    buffer.clear()
    if markdown:
        elements.append(_build_markdown_element(markdown))


def _build_markdown_element(markdown: str) -> dict[str, str]:
    return {
        "tag": "markdown",
        "content": markdown.strip() or " ",
        "text_align": "left",
        "text_size": "normal_v2",
    }


def _render_latex_for_feishu(markdown: str) -> str:
    parts: list[str] = []
    buffer: list[str] = []
    in_code_block = False

    def flush_buffer() -> None:
        if not buffer:
            return
        parts.append(_render_latex_in_plain_markdown("".join(buffer)))
        buffer.clear()

    for line in markdown.splitlines(keepends=True):
        if line.strip().startswith("```"):
            was_in_code_block = in_code_block
            if not was_in_code_block:
                flush_buffer()
            parts.append(line)
            in_code_block = not was_in_code_block
            continue

        if in_code_block:
            parts.append(line)
        else:
            buffer.append(line)

    flush_buffer()
    return "".join(parts)


def _render_latex_in_plain_markdown(text: str) -> str:
    rendered = _latex_block_pattern.sub(_replace_latex_block_match, text)
    rendered = _latex_standalone_bracket_block_pattern.sub(_replace_latex_match, rendered)
    for pattern in (_latex_paren_pattern, _latex_dollar_pattern, _latex_bare_bracket_pattern):
        rendered = pattern.sub(_replace_latex_match, rendered)
    return rendered


def _replace_latex_block_match(match: re.Match[str]) -> str:
    return _latex_to_readable_text(match.group("expr").strip())


def _replace_latex_match(match: re.Match[str]) -> str:
    expr = match.group("expr").strip()
    if not _looks_like_latex_math(expr):
        return match.group(0)
    return _latex_to_readable_text(expr)


def _looks_like_latex_math(expr: str) -> bool:
    return bool(
        re.search(
            r"\\(?:frac|text|times|div|cdot|leq?|geq?|neq|approx|left|right|sum|sqrt)(?![A-Za-z])|[_^{}]",
            expr,
        )
    )


def _replace_latex_fractions(text: str) -> str:
    parts: list[str] = []
    index = 0

    while index < len(text):
        if not text.startswith(r"\frac", index):
            parts.append(text[index])
            index += 1
            continue

        command_end = index + len(r"\frac")
        if command_end < len(text) and text[command_end].isascii() and text[command_end].isalpha():
            parts.append(text[index])
            index += 1
            continue

        numerator_start = _skip_latex_whitespace(text, command_end)
        numerator, after_numerator = _read_latex_group(text, numerator_start)
        if numerator is None:
            parts.append(" / ")
            index = command_end
            continue

        denominator_start = _skip_latex_whitespace(text, after_numerator)
        denominator, after_denominator = _read_latex_group(text, denominator_start)
        if denominator is None:
            parts.append(f"({_latex_to_readable_text(numerator)}) / ")
            index = after_numerator
            continue

        parts.append(
            f"({_latex_to_readable_text(numerator)}) / ({_latex_to_readable_text(denominator)})"
        )
        index = after_denominator

    return "".join(parts)


def _skip_latex_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _read_latex_group(text: str, index: int) -> tuple[str | None, int]:
    if index >= len(text) or text[index] != "{":
        return None, index

    depth = 1
    cursor = index + 1
    start = cursor

    while cursor < len(text):
        char = text[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:cursor], cursor + 1
        cursor += 1

    return None, index


def _latex_to_readable_text(expr: str) -> str:
    text = expr.strip()
    text = _replace_latex_fractions(text)

    text = re.sub(r"\\(?:text|mathbf|mathrm|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace(r"\left", "").replace(r"\right", "")
    replacements = {
        r"\ ": " ",
        r"\,": " ",
        r"\;": " ",
        r"\:": " ",
        r"\times": " x ",
        r"\cdot": " * ",
        r"\div": " / ",
        r"\leq": " <= ",
        r"\le": " <= ",
        r"\geq": " >= ",
        r"\ge": " >= ",
        r"\neq": " != ",
        r"\approx": " ~= ",
        r"\%": "%",
        r"\$": "$",
        r"\&": "&",
        r"\_": "_",
    }
    for latex, plain in replacements.items():
        text = text.replace(latex, plain)

    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    protected_operators = {
        "<=": "__LATEX_LE__",
        ">=": "__LATEX_GE__",
        "!=": "__LATEX_NE__",
        "~=": "__LATEX_APPROX__",
    }
    for operator, placeholder in protected_operators.items():
        text = text.replace(operator, placeholder)
    text = re.sub(r"\s*([=+\-*<>])\s*", r" \1 ", text)
    for operator, placeholder in protected_operators.items():
        text = text.replace(placeholder, f" {operator} ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


def _rewrite_reference_links(markdown: str) -> tuple[str, list[_ReferenceSource]]:
    markdown = _strip_model_generated_source_section(markdown)
    references_by_url: dict[str, _ReferenceSource] = {}
    in_code_block = False

    def register_reference(url: str, label: str, description: str) -> _ReferenceSource:
        nonlocal references_by_url
        normalized_url = _normalize_reference_url(url)
        if normalized_url not in references_by_url:
            references_by_url[normalized_url] = _ReferenceSource(
                number=len(references_by_url) + 1,
                label=label,
                description=description,
                url=url,
            )
        return references_by_url[normalized_url]

    def format_short_reference(source: _ReferenceSource) -> str:
        return f"[[{source.number}]]({_format_markdown_link_url(source.url)})"

    def replace_reference_tag(match: re.Match[str]) -> str:
        if in_code_block:
            return match.group(0)

        raw_path = match.group(1).strip()
        if not raw_path:
            return ""

        description = _extract_reference_file_name(raw_path) or raw_path
        source = register_reference(raw_path, f"引用{len(references_by_url) + 1}", description)
        return format_short_reference(source)

    def replace_match(match: re.Match[str]) -> str:
        if in_code_block:
            return match.group(0)

        label = match.group(1).strip()
        url, existing_title = _parse_markdown_link_destination(match.group(2))
        reference_match = _reference_link_label_pattern.match(label)
        if not reference_match and existing_title:
            reference_match = _reference_link_label_pattern.match(existing_title)
        if not reference_match:
            return match.group(0)

        label_text = reference_match.group(1).replace(" ", "")
        description = existing_title or label
        source = register_reference(url, label_text, description)
        return format_short_reference(source)

    rewritten_lines: list[str] = []
    for line in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            rewritten_lines.append(line)
            continue
        rewritten_line = _reference_tag_pattern.sub(replace_reference_tag, line)
        rewritten_line = _markdown_link_pattern.sub(replace_match, rewritten_line)
        rewritten_lines.append(_dedupe_adjacent_reference_links(rewritten_line))

    rewritten_markdown = _join_adjacent_reference_links("\n".join(rewritten_lines))
    return rewritten_markdown.strip(), list(references_by_url.values())


def _dedupe_adjacent_reference_links(markdown: str) -> str:
    previous = None
    current = markdown
    while current != previous:
        previous = current
        current = _adjacent_reference_pattern.sub(r"\g<ref>", current)
    return current


def _join_adjacent_reference_links(markdown: str) -> str:
    parts: list[str] = []
    buffer: list[str] = []
    in_code_block = False

    def flush_buffer() -> None:
        if not buffer:
            return
        current = "".join(buffer)
        previous = None
        while current != previous:
            previous = current
            current = _adjacent_any_reference_separator_pattern.sub(r"\g<ref>", current)
        parts.append(current)
        buffer.clear()

    for line in markdown.splitlines(keepends=True):
        if line.strip().startswith("```"):
            was_in_code_block = in_code_block
            if not was_in_code_block:
                flush_buffer()
            parts.append(line)
            in_code_block = not was_in_code_block
            continue

        if in_code_block:
            parts.append(line)
        else:
            buffer.append(line)

    flush_buffer()
    return "".join(parts)


def _parse_markdown_link_destination(raw_destination: str) -> tuple[str, str]:
    destination = raw_destination.strip()
    title_match = _markdown_link_title_pattern.match(destination)
    if not title_match:
        return destination, ""
    return title_match.group("url").strip(), title_match.group("title").strip()


def _strip_model_generated_source_section(markdown: str) -> str:
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    in_code_block = False
    cutoff: int | None = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if _source_section_heading_pattern.match(stripped):
            cutoff = _trim_source_section_separator(lines, index)
            break

    if cutoff is None:
        return markdown
    return "\n".join(lines[:cutoff]).rstrip()


def _trim_source_section_separator(lines: list[str], source_index: int) -> int:
    index = source_index
    while index > 0 and not lines[index - 1].strip():
        index -= 1
    if index > 0 and re.fullmatch(r"[-*_]\s*[-*_]\s*[-*_\s]*", lines[index - 1].strip()):
        index -= 1
    return index


def _normalize_reference_url(url: str) -> str:
    return url.replace("\\", "/").strip()


def _format_markdown_link_url(url: str) -> str:
    if _is_lark_document_url(url):
        return url
    lark_url = _resolve_lark_document_url(url)
    if lark_url:
        return lark_url
    if _is_document_download_url(url):
        return url
    return _build_document_download_url(url)


def _is_lark_document_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    return host.endswith("feishu.cn") or host.endswith("larksuite.com")


def _is_document_download_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    return parsed.path.rstrip("/").endswith("/api/v1/documents/download")


def _resolve_lark_document_url(raw_path: str) -> str | None:
    file_name = _extract_reference_file_name(raw_path)
    if not file_name:
        return None

    mapping = _load_local_to_lark_mapping()
    return mapping.get(_normalize_document_name(file_name))


def _extract_reference_file_name(raw_path: str) -> str:
    normalized_path = unquote(_normalize_reference_url(raw_path)).strip().strip("\"'")
    if not normalized_path:
        return ""

    parsed = urlparse(normalized_path)
    query_path = ""
    if parsed.query:
        query_path = (parse_qs(parsed.query).get("path") or [""])[0]

    candidate = unquote(query_path or parsed.path or normalized_path)
    candidate = candidate.replace("\\", "/").strip().strip("\"'")
    return PurePosixPath(candidate).name


def _normalize_document_name(file_name: str) -> str:
    return re.sub(r"\s+", "", file_name).strip().lower()


def _load_local_to_lark_mapping() -> dict[str, str]:
    global _local_to_lark_mapping_cache

    if _local_to_lark_mapping_cache is not None:
        return _local_to_lark_mapping_cache

    mapping: dict[str, str] = {}
    if not _local_to_lark_mapping_dir.exists():
        _local_to_lark_mapping_cache = mapping
        return mapping

    for mapping_path in sorted(_local_to_lark_mapping_dir.glob("*.json")):
        try:
            with mapping_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            _warn(
                "local-to-lark mapping ignored because it cannot be loaded",
                path=str(mapping_path),
                error=str(exc),
            )
            continue

        if not isinstance(data, dict):
            _warn(
                "local-to-lark mapping ignored because root value is not an object",
                path=str(mapping_path),
            )
            continue

        for file_name, lark_url in data.items():
            if not isinstance(file_name, str) or not isinstance(lark_url, str):
                continue
            key = _normalize_document_name(file_name)
            if key and key not in mapping:
                mapping[key] = lark_url

    _local_to_lark_mapping_cache = mapping
    return mapping


def _build_document_download_url(raw_path: str) -> str:
    base_url = settings.public_base_url or "http://localhost:8000"
    normalized_path = _normalize_minio_reference_for_download(raw_path)
    query = urlencode({"path": normalized_path}, quote_via=quote)
    return f"{base_url.rstrip('/')}/api/v1/documents/download?{query}"


def _normalize_minio_reference_for_download(raw_path: str) -> str:
    normalized_path = _normalize_reference_url(raw_path)
    parsed = urlparse(normalized_path)
    if parsed.query:
        query_path = (parse_qs(parsed.query).get("path") or [""])[0]
        if query_path:
            normalized_path = query_path
    try:
        return parse_raw_document_reference(normalized_path).uri
    except ValueError:
        return normalized_path


def _build_reference_sources_panel(sources: list[_ReferenceSource]) -> dict[str, Any]:
    source_lines: list[str] = []
    for source in sources:
        description = source.description.strip() or source.label
        source_lines.append(f"[[{source.number}]]({_format_markdown_link_url(source.url)}) {description}")

    return {
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {
            "title": {
                "tag": "plain_text",
                "content": "知识来源",
            }
        },
        "elements": [_build_markdown_element("\n".join(source_lines))],
    }


def _normalize_card_markdown(markdown: str) -> str:
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized_lines: list[str] = []
    in_code_block = False
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            normalized_lines.append(line)
            index += 1
            continue

        if in_code_block:
            normalized_lines.append(line)
            index += 1
            continue

        table_end = _find_markdown_table_end(lines, index)
        if table_end is not None:
            normalized_lines.extend(_markdown_table_to_card_lines(lines[index:table_end]))
            index = table_end
            continue

        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            normalized_lines.extend(_heading_to_card_markdown(level, title))
            index += 1
            continue

        if _is_markdown_horizontal_rule(stripped):
            normalized_lines.append("────────")
            index += 1
            continue

        if stripped.startswith(">"):
            normalized_lines.append(_normalize_blockquote_line(line))
            index += 1
            continue

        normalized_lines.append(_normalize_card_markdown_line(line))
        index += 1

    return "\n".join(normalized_lines).strip() or " "


def _find_markdown_table_end(lines: list[str], start_index: int) -> int | None:
    if start_index + 1 >= len(lines):
        return None
    header = lines[start_index].strip()
    separator = lines[start_index + 1].strip()
    if not _looks_like_markdown_table_row(header):
        return None
    if not _is_markdown_table_separator(separator):
        return None

    end = start_index + 2
    while end < len(lines) and _looks_like_markdown_table_row(lines[end].strip()):
        end += 1
    return end


def _looks_like_markdown_table_row(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def _is_markdown_table_separator(line: str) -> bool:
    if not _looks_like_markdown_table_row(line):
        return False
    cells = _split_markdown_table_row(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _split_markdown_table_row(line: str) -> list[str]:
    trimmed = line.strip().strip("|")
    return [cell.strip() for cell in trimmed.split("|")]


def _markdown_table_to_card_lines(table_lines: list[str]) -> list[str]:
    headers = _split_markdown_table_row(table_lines[0])
    rows = [_split_markdown_table_row(line) for line in table_lines[2:]]
    normalized_rows: list[str] = []

    for row_index, row in enumerate(rows, start=1):
        if not any(cell.strip() for cell in row):
            continue
        pairs = []
        for column_index, header in enumerate(headers):
            value = row[column_index].strip() if column_index < len(row) else ""
            if not value:
                continue
            label = header.strip() or f"列{column_index + 1}"
            pairs.append(f"{label}: {value}")
        if pairs:
            normalized_rows.append(f"{row_index}. " + "；".join(pairs))

    return normalized_rows or ["；".join(headers)]


def _is_markdown_horizontal_rule(stripped_line: str) -> bool:
    return bool(re.fullmatch(r"[-*_]\s*[-*_]\s*[-*_\s]*", stripped_line)) and len(
        stripped_line.replace(" ", "")
    ) >= 3


def _normalize_blockquote_line(line: str) -> str:
    quote = re.sub(r"^\s*>\s?", "", line).strip()
    return f"▌ {quote}" if quote else "▌"


def _normalize_card_markdown_line(line: str) -> str:
    list_item = re.match(r"^(\s*)[*+-]\s+(.+?)\s*$", line)
    if list_item:
        indent, content = list_item.groups()
        return f"{indent}· {_normalize_list_item_content(content)}"

    return line


def _normalize_list_item_content(content: str) -> str:
    italic = re.match(r"^\*(.+?)\*\s*$", content)
    if italic and "来源：" in italic.group(1):
        return italic.group(1)
    return content


def _heading_to_card_markdown(level: int, title: str) -> list[str]:
    if level == 1:
        return [f"**{title}**", "---"]
    if level == 2:
        return ["", f"**{title}**"]
    if level == 3:
        return ["", f"**{title}**"]
    return [f"**{title}**"]


def _link_tag_to_markdown(raw_link: str) -> str:
    link = raw_link.strip()
    if not link:
        return ""
    return f"[{link}]({link})"


async def _upload_local_image(raw_path: str, token: str) -> str | None:
    image_path = _resolve_local_image_path(raw_path)
    if image_path is None:
        _warn("image ignored because local file does not exist", path=raw_path)
        return None

    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    url = f"{settings.feishu_base_url}/open-apis/im/v1/images"
    timeout = httpx.Timeout(settings.feishu_timeout)

    _debug("image upload request", path=str(image_path), url=url, mime_type=mime_type)
    try:
        with image_path.open("rb") as image_file:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    data={"image_type": "message"},
                    files={"image": (image_path.name, image_file, mime_type)},
                )
        _debug(
            "image upload response",
            path=str(image_path),
            http_status=response.status_code,
            body_preview=_preview(response.text),
        )
        response.raise_for_status()
    except (OSError, httpx.HTTPError) as exc:
        _warn("image ignored because upload failed", path=str(image_path), error=str(exc))
        return None

    try:
        result = response.json()
    except ValueError:
        _warn("image ignored because upload response is not json", path=str(image_path))
        return None

    image_key = (result.get("data") or {}).get("image_key")
    if result.get("code") != 0 or not image_key:
        _warn("image ignored because upload API returned error", path=str(image_path), response=result)
        return None

    _debug("image upload success", path=str(image_path), image_key=image_key)
    return str(image_key)


def _resolve_local_image_path(raw_path: str) -> Path | None:
    cleaned_path = raw_path.strip().strip("\"'")
    if not cleaned_path:
        return None

    normalized_path = cleaned_path.replace("\\", "/")
    candidates = [Path(normalized_path)]
    if not Path(normalized_path).is_absolute():
        candidates.append(Path.cwd() / normalized_path)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


async def _get_tenant_access_token() -> str:
    global _tenant_access_token, _tenant_access_token_expires_at

    if _tenant_access_token and time.time() < _tenant_access_token_expires_at:
        _debug("tenant token cache hit")
        return _tenant_access_token

    timeout = httpx.Timeout(settings.feishu_timeout)
    url = f"{settings.feishu_base_url}/open-apis/auth/v3/tenant_access_token/internal"
    _debug(
        "tenant token request",
        url=url,
        has_app_id=bool(settings.feishu_app_id),
        has_app_secret=bool(settings.feishu_app_secret),
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json={
                    "app_id": settings.feishu_app_id,
                    "app_secret": settings.feishu_app_secret,
                },
            )
            _debug(
                "tenant token response",
                http_status=response.status_code,
                has_token=_response_has_tenant_token(response),
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Feishu token API timed out") from exc
    except httpx.HTTPStatusError as exc:
        detail = _response_text(exc.response)
        raise HTTPException(
            status_code=502,
            detail=f"Feishu token API returned {exc.response.status_code}: {detail}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to call Feishu token API: {exc}",
        ) from exc

    result = response.json()
    token = result.get("tenant_access_token")
    if result.get("code") != 0 or not token:
        raise HTTPException(status_code=502, detail=f"Feishu token API error: {result}")

    expire_seconds = int(result.get("expire") or 7200)
    _tenant_access_token = token
    _tenant_access_token_expires_at = time.time() + max(expire_seconds - 120, 60)
    _debug("tenant token success", expire_seconds=expire_seconds)
    return token


def _response_text(response: httpx.Response) -> str:
    text = response.text.strip()
    return text[:500] if text else "<empty response>"


def _response_has_tenant_token(response: httpx.Response) -> bool:
    try:
        payload = response.json()
    except ValueError:
        return False
    return bool(payload.get("tenant_access_token"))


def _detail_to_text(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    return json.dumps(detail, ensure_ascii=False, default=str)


def _is_feishu_card_content_error(exc: HTTPException) -> bool:
    detail = _detail_to_text(exc.detail)
    return "Failed to create card content" in detail or "unsupported tag" in detail


def _preview(value: Any, max_length: int = 160) -> str:
    text = str(value).replace("\n", "\\n")
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}..."


def _debug(message: str, **fields: Any) -> None:
    field_text = " ".join(
        f"{key}={json.dumps(value, ensure_ascii=False, default=str)}"
        for key, value in fields.items()
    )
    suffix = f" {field_text}" if field_text else ""
    print(f"[feishu] {message}{suffix}", flush=True)


def _warn(message: str, **fields: Any) -> None:
    field_text = " ".join(
        f"{key}={json.dumps(value, ensure_ascii=False, default=str)}"
        for key, value in fields.items()
    )
    suffix = f" {field_text}" if field_text else ""
    print(f"[feishu][warn] {message}{suffix}", flush=True)
