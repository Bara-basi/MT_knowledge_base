from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import mimetypes
import re
import time
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
from app.schemas.query import QueryRequest
from app.services.chat_records import create_chat_record, record_chat_answer
from app.services.evaluate.evaluate import evaluate_answer_fallback
from app.services.llm import LLMAPIError, LLMConfigError, LLMClient, LLMSettings


router = APIRouter(prefix="/feishu", tags=["feishu"])
logger = logging.getLogger(__name__)

_tenant_access_token: str | None = None
_tenant_access_token_expires_at = 0.0
_seen_message_keys: dict[str, float] = {}
_seen_message_ttl_seconds = 3600
_pseudo_tag_pattern = re.compile(r"<(img|link)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_markdown_link_pattern = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)")
_markdown_link_title_pattern = re.compile(r"^(?P<url>.+?)\s+\"(?P<title>[^\"]*)\"\s*$")
_reference_link_label_pattern = re.compile(r"^\s*(片段\s*\d+)\s*[,，:：]\s*(.+?)\s*$")
_adjacent_reference_pattern = re.compile(
    r"(?P<ref>\[\[(?P<number>\d+)\]\]\([^)]+\))(?P<separator>\s*)(?P=ref)"
)
_source_section_heading_pattern = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?(知识来源|引用文献|参考来源)(?:\*\*)?\s*[:：]?\s*$"
)
_feedback_interval_seconds = 1.5
_comfort_feedback_interval_seconds = 60.0
_stream_update_min_interval_seconds = 0.25
_pseudo_stream_update_delay_seconds = 0.12
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

_n8n_progress_texts = {
    "understanding": "🤔 正在理解问题...",
    "retrieving": "🔍 正在检索知识库...",
    "reranking": "📚 正在整理资料...",
    "generating": "✍️ 正在组织答案...",
}
_n8n_progress_stage_order = ("understanding", "retrieving", "reranking", "generating")

# n8n execution data is only reliable after a node appears in runData.
# These mappings intentionally treat a completed node as the signal for the next stage.
_n8n_completed_node_stage_ids = {
    "understanding": set(),
    "retrieving": {
        "54aad033-e2d6-4b2a-aa73-027f9fc839ce",
    },
    "reranking": {
        "53fb4814-a531-4422-b6ba-f38c3db5a9a4",
        "18cbcf26-ca22-443f-aa09-3c4173e60983",
        "a509efe4-649b-4cf7-b32a-5ad35a45a60a",
        "8a620c5f-620a-481a-96f1-880f595ebb74",
        "db342e62-4de8-486a-9443-ddfafc679b77",
    },
    "generating": {
        "6113024b-7803-4ce2-930a-c893ff1ff7fd",
    },
}
_n8n_completed_node_stage_names = {
    "understanding": set(),
    "retrieving": {"query转写", "query_rewrite"},
    "reranking": {
        "前往问答检索",
        "前往问答检索1",
        "前往问答检索2",
        "前往问答检索3",
        "Code in JavaScript",
    },
    "generating": {"合并item并去重"},
}

# Current-node mappings are kept as a fallback for long-running nodes that are visible mid-run.
_n8n_current_node_stage_ids = {
    "understanding": set(),
    "retrieving": {
        "c3d63d86-f11f-4e59-a65e-d6d12c630a5f",
        "4370e0b8-88ff-459f-8625-5a48e20edb06",
    },
    "reranking": {"be8a0486-cbf0-4b6e-95c2-9dc978abcbc7"},
    "generating": {"8ab6c4c9-2d7a-4835-b80b-1caadbca9561"},
}
_n8n_current_node_stage_names = {
    "understanding": set(),
    "retrieving": {"When Executed by Another Workflow", "HTTP Request"},
    "reranking": {"合并多路召回结果"},
    "generating": {"检索回答", "检索问答"},
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
        _debug("event ignored", reason="empty_or_non_text_message", message_id=message_id)
        return {"ok": True, "ignored": True, "reason": "empty_or_non_text_message"}

    background_tasks.add_task(
        _answer_feishu_message,
        question=question,
        sender_id=_extract_sender_id(event.get("sender") or {}),
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
    feedback_handle_task = _schedule_initial_feishu_feedback(
        question=question,
        chat_id=chat_id,
        message_id=message_id,
        event_id=event_id,
        dedupe_key=dedupe_key,
        n8n_started_after=n8n_started_after,
    )

    try:
        await create_chat_record(
            user_id=sender_id,
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
            return

        try:
            await record_chat_answer(
                user_id=sender_id,
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
        return

    feedback_handle = await _resolve_initial_feishu_feedback(feedback_handle_task, wait=False)
    await _stop_feishu_feedback_loop(feedback_handle)

    if not message_id:
        _debug(
            "reply skipped",
            reason="missing_message_id",
            event_id=event_id,
            dedupe_key=dedupe_key,
        )
        return

    if feedback_handle:
        sent = await _pseudo_stream_feishu_final_answer(
            message_id=feedback_handle.message_id,
            answer=response.answer,
        )
    else:
        sent_message_id = await _pseudo_stream_new_feishu_final_answer(
            chat_id=chat_id,
            reply_to_message_id=message_id,
            answer=response.answer,
        )
        sent = sent_message_id is not None
    _debug(
        "reply finished",
        event_id=event_id,
        message_id=message_id,
        dedupe_key=dedupe_key,
        sent=sent,
    )
    _schedule_feishu_answer_evaluation(
        question=question,
        answer=response.answer,
        sender_id=sender_id,
        chat_id=chat_id,
        event_id=event_id,
        message_id=message_id,
        dedupe_key=dedupe_key,
    )


def _schedule_feishu_answer_evaluation(
    *,
    question: str,
    answer: str,
    sender_id: str | None,
    chat_id: str | None,
    event_id: str | None,
    message_id: str | None,
    dedupe_key: str | None,
) -> None:
    task = asyncio.create_task(
        _run_feishu_answer_evaluation(
            question=question,
            answer=answer,
            sender_id=sender_id,
            chat_id=chat_id,
            event_id=event_id,
            message_id=message_id,
            dedupe_key=dedupe_key,
        )
    )
    task.add_done_callback(_log_feishu_answer_evaluation_task_failure)


def _log_feishu_answer_evaluation_task_failure(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as exc:  # noqa: BLE001 - keep background task failures visible.
        logger.exception("Unexpected Feishu answer evaluation task failure: %s", exc)


async def _run_feishu_answer_evaluation(
    *,
    question: str,
    answer: str,
    sender_id: str | None,
    chat_id: str | None,
    event_id: str | None,
    message_id: str | None,
    dedupe_key: str | None,
) -> None:
    try:
        fallback_result = await asyncio.to_thread(
            evaluate_answer_fallback,
            question=question,
            answer=answer,
            user_id=sender_id,
            session_id=chat_id,
            conversation_id=chat_id,
            reference_answer=None,
            persist=True,
            verbose=False,
        )
        _debug(
            "fallback evaluation recorded",
            event_id=event_id,
            message_id=message_id,
            dedupe_key=dedupe_key,
            fallback=fallback_result.get("fallback"),
            reason=_preview(fallback_result.get("reason", "")),
        )
    except Exception as exc:  # noqa: BLE001 - evaluation must not block user replies.
        logger.exception("Failed to evaluate fallback for Feishu message: %s", exc)
        _warn(
            "fallback evaluation failed",
            event_id=event_id,
            message_id=message_id,
            dedupe_key=dedupe_key,
            error=str(exc),
        )


def _schedule_initial_feishu_feedback(
    *,
    question: str,
    chat_id: str | None,
    message_id: str | None,
    event_id: str | None,
    dedupe_key: str | None,
    n8n_started_after: float,
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
) -> _FeishuFeedbackHandle | None:
    streamed_initial = await _send_streaming_initial_feedback_card(
        question=question,
        chat_id=chat_id,
        message_id=message_id,
    )
    if streamed_initial is None:
        _debug(
            "initial feedback skipped",
            reason="greeting_model_returned_null",
            event_id=event_id,
            message_id=message_id,
            dedupe_key=dedupe_key,
        )
        return None
    feedback_message_id, prefix_text, status_text = streamed_initial

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
    return _FeishuFeedbackHandle(
        state=feedback_state,
        stop_event=stop_event,
        task=feedback_task,
        comfort_task=comfort_task,
    )


async def _send_streaming_initial_feedback_card(
    *,
    question: str,
    chat_id: str | None,
    message_id: str | None,
) -> tuple[str | None, str, str] | None:
    status_text = _get_initial_feedback_text()
    prefix_buffer = ""
    feedback_message_id: str | None = None
    last_update_at = 0.0

    try:
        async for chunk in _stream_immediate_feedback_greeting(question):
            prefix_buffer += chunk
            prefix_text = _clean_immediate_feedback_greeting(prefix_buffer)
            if prefix_text is None:
                continue
            if not prefix_text:
                continue

            if feedback_message_id is None:
                feedback_message_id = await _send_initial_feedback_card(
                    chat_id=chat_id,
                    message_id=message_id,
                    markdown_text=_compose_feedback_card_text(prefix_text, status_text),
                )
                last_update_at = time.monotonic()
                continue

            now = time.monotonic()
            if now - last_update_at >= _stream_update_min_interval_seconds:
                await _try_update_feishu_markdown_message(
                    feedback_message_id,
                    _compose_feedback_card_text(prefix_text, status_text),
                    log_content=False,
                )
                last_update_at = now

        final_prefix = _clean_immediate_feedback_greeting(prefix_buffer)
        if final_prefix is None:
            return None
        if not final_prefix:
            raise LLMAPIError("Immediate feedback stream returned empty text")
        if feedback_message_id is None:
            feedback_message_id = await _send_initial_feedback_card(
                chat_id=chat_id,
                message_id=message_id,
                markdown_text=_compose_feedback_card_text(final_prefix, status_text),
            )
        else:
            await _try_update_feishu_markdown_message(
                feedback_message_id,
                _compose_feedback_card_text(final_prefix, status_text),
                log_content=False,
            )
        return feedback_message_id, final_prefix, status_text
    except (asyncio.TimeoutError, LLMAPIError, LLMConfigError) as exc:
        _warn("initial feedback stream fallback", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - initial feedback should stay best-effort.
        logger.exception("Failed to stream initial Feishu greeting: %s", exc)

    initial_parts = await _build_initial_feedback_parts(question)
    if initial_parts is None:
        return None
    prefix_text, status_text = initial_parts
    feedback_message_id = await _send_initial_feedback_card(
        chat_id=chat_id,
        message_id=message_id,
        markdown_text=_compose_feedback_card_text(prefix_text, status_text),
    )
    return feedback_message_id, prefix_text, status_text


async def _send_initial_feedback_card(
    *,
    chat_id: str | None,
    message_id: str | None,
    markdown_text: str,
) -> str | None:
    if message_id:
        feedback_message_id = await _try_send_feishu_markdown_message(
            chat_id=chat_id,
            reply_to_message_id=message_id,
            markdown_text=markdown_text,
            log_content=False,
        )
        if feedback_message_id:
            return feedback_message_id
        return await _try_send_feishu_markdown_reply(
            message_id,
            markdown_text,
            log_content=False,
        )
    if chat_id:
        return await _try_send_feishu_markdown_message(
            chat_id=chat_id,
            reply_to_message_id=None,
            markdown_text=markdown_text,
            log_content=False,
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


async def _stream_immediate_feedback_greeting(question: str):
    messages = _build_immediate_feedback_messages(question)
    async for chunk in _stream_openai_chat_completion(
        messages=messages,
        model=settings.immediate_feedback_model,
        temperature=0.3,
        max_tokens=settings.immediate_feedback_max_tokens,
    ):
        yield chunk


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


async def _stream_openai_chat_completion(
    *,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
):
    if not settings.siliconflow_api_key:
        raise LLMConfigError("Missing SILICONFLOW_API_KEY for streaming feedback")

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    payload.update(_build_immediate_feedback_extra_body())
    timeout = httpx.Timeout(
        timeout=settings.immediate_feedback_timeout,
        connect=settings.immediate_feedback_connect_timeout,
        read=settings.immediate_feedback_timeout,
        write=settings.immediate_feedback_timeout,
        pool=min(settings.immediate_feedback_timeout, 1.0),
    )
    headers = {
        "Authorization": f"Bearer {settings.siliconflow_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{settings.siliconflow_base_url}/chat/completions"

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[len("data:") :].strip()
                if not line:
                    continue
                if line == "[DONE]":
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        yield content


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
            updated = await _stream_comfort_feedback_prefix(
                state=state,
                minute=minute,
            )
            if not updated:
                await _update_feishu_feedback_prefix(
                    state,
                    _get_local_comfort_feedback_text(minute, state.status_text),
                )
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


async def _stream_comfort_feedback_prefix(
    *,
    state: _FeishuFeedbackState,
    minute: int,
) -> bool:
    buffer = ""
    last_update_at = 0.0
    try:
        async for chunk in _stream_comfort_feedback_text(
            question=state.question,
            minute=minute,
            status_text=state.status_text,
        ):
            buffer += chunk
            cleaned = _clean_comfort_feedback_text(buffer)
            if not cleaned:
                continue
            now = time.monotonic()
            if now - last_update_at >= _stream_update_min_interval_seconds:
                await _update_feishu_feedback_prefix(state, cleaned)
                last_update_at = now
        final_text = _clean_comfort_feedback_text(buffer)
        if final_text:
            await _update_feishu_feedback_prefix(state, final_text)
            return True
    except (asyncio.TimeoutError, LLMAPIError, LLMConfigError) as exc:
        _warn("comfort feedback stream fallback", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - comfort feedback must stay best-effort.
        logger.exception("Failed to stream comfort feedback text: %s", exc)
    return False


async def _stream_comfort_feedback_text(
    *,
    question: str,
    minute: int,
    status_text: str,
):
    messages = _build_comfort_feedback_messages(
        question=question,
        minute=minute,
        status_text=status_text,
    )
    async for chunk in _stream_openai_chat_completion(
        messages=messages,
        model=settings.immediate_feedback_model,
        temperature=0.45,
        max_tokens=min(settings.immediate_feedback_max_tokens, 60),
    ):
        yield chunk


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
        return await _try_update_feishu_markdown_message(
            state.message_id,
            _compose_feedback_card_text(state.prefix_text, state.status_text),
            log_content=False,
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

    try:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
                break
            except asyncio.TimeoutError:
                pass

            stage = await _poll_n8n_progress_stage(started_after=started_after)
            if stage and _n8n_stage_index(stage) > _n8n_stage_index(last_stage):
                last_stage = stage
                text = _n8n_progress_texts[stage]
                if text != last_text:
                    last_text = text
                    await _update_feishu_feedback_status(state, text)
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
            executions.extend(
                await _list_recent_n8n_executions(
                    client=client,
                    base_url=base_url,
                    workflow_id=workflow_id,
                    started_after=started_after,
                )
            )

        stage: str | None = None
        for execution in sorted(
            executions,
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
) -> list[dict[str, Any]]:
    response = await client.get(
        f"{base_url}/api/v1/executions",
        params={"workflowId": workflow_id, "limit": 10, "includeData": "false"},
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
        if node_name in _n8n_completed_node_stage_names[stage]:
            return stage
    for stage in _n8n_progress_stage_order:
        if node_id_text and node_id_text in _n8n_current_node_stage_ids[stage]:
            return stage
        if node_name in _n8n_current_node_stage_names[stage]:
            return stage
    return None


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


async def _try_send_feishu_markdown_message(
    *,
    chat_id: str | None,
    reply_to_message_id: str | None,
    markdown_text: str,
    log_content: bool = True,
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
            )
        except HTTPException as exc:
            detail = _detail_to_text(exc.detail)
            logger.exception("Failed to send Feishu markdown card to chat %s: %s", chat_id, detail)
            _debug("markdown send failed", chat_id=chat_id, error=detail)

    if reply_to_message_id:
        return await _try_send_feishu_markdown_reply(
            reply_to_message_id,
            markdown_text,
            log_content=log_content,
        )

    return None


async def _try_send_feishu_markdown_reply(
    message_id: str,
    markdown_text: str,
    *,
    log_content: bool = True,
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
        )
    except HTTPException as exc:
        detail = _detail_to_text(exc.detail)
        logger.exception("Failed to send Feishu markdown card %s: %s", message_id, detail)
        _debug("markdown reply failed", message_id=message_id, error=detail)
        return None


async def _try_reply_feishu_markdown(message_id: str, markdown_text: str) -> bool:
    return await _try_send_feishu_markdown_reply(message_id, markdown_text) is not None


async def _pseudo_stream_feishu_final_answer(
    *,
    message_id: str,
    answer: str,
) -> bool:
    chunks = _build_pseudo_stream_chunks(answer)
    if not chunks:
        return await _try_update_feishu_markdown_message(message_id, answer)

    sent = True
    for index, chunk in enumerate(chunks):
        sent = await _try_update_feishu_markdown_message(
            message_id,
            chunk,
            log_content=index == len(chunks) - 1,
        )
        if not sent:
            return False
        if index < len(chunks) - 1:
            await asyncio.sleep(_pseudo_stream_update_delay_seconds)
    return sent


async def _pseudo_stream_new_feishu_final_answer(
    *,
    chat_id: str | None,
    reply_to_message_id: str | None,
    answer: str,
) -> str | None:
    chunks = _build_pseudo_stream_chunks(answer)
    if not chunks:
        return await _try_send_feishu_markdown_message(
            chat_id=chat_id,
            reply_to_message_id=reply_to_message_id,
            markdown_text=answer,
        )

    sent_message_id = await _try_send_feishu_markdown_message(
        chat_id=chat_id,
        reply_to_message_id=reply_to_message_id,
        markdown_text=chunks[0],
        log_content=False,
    )
    if not sent_message_id:
        return None

    for index, chunk in enumerate(chunks[1:], start=1):
        await asyncio.sleep(_pseudo_stream_update_delay_seconds)
        updated = await _try_update_feishu_markdown_message(
            sent_message_id,
            chunk,
            log_content=index == len(chunks) - 1,
        )
        if not updated:
            return sent_message_id
    return sent_message_id


def _build_pseudo_stream_chunks(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= 160:
        return [cleaned]

    chunks: list[str] = []
    step = 120
    index = step
    while index < len(cleaned):
        boundary = _find_stream_chunk_boundary(cleaned, index, window=40)
        chunks.append(cleaned[:boundary].rstrip())
        index = boundary + step
    if not chunks or chunks[-1] != cleaned:
        chunks.append(cleaned)
    return chunks


def _find_stream_chunk_boundary(text: str, target: int, *, window: int) -> int:
    end = min(len(text), target + window)
    start = max(0, target - window)
    for index in range(min(end, len(text) - 1), start, -1):
        if text[index] in "。！？；\n，,.;!?":
            return index + 1
    return min(target, len(text))


async def _try_update_feishu_markdown_message(
    message_id: str,
    markdown_text: str,
    *,
    log_content: bool = True,
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
        )
    except HTTPException as exc:
        detail = _detail_to_text(exc.detail)
        logger.exception("Failed to update Feishu markdown card %s: %s", message_id, detail)
        _debug("markdown update failed", message_id=message_id, error=detail)
        return False


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
    if message.get("message_type") != "text":
        return ""

    content = message.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return content.strip()

    if isinstance(content, dict):
        text = content.get("text")
        return text.strip() if isinstance(text, str) else ""

    return ""


def _extract_sender_id(sender: dict[str, Any]) -> str | None:
    sender_id = sender.get("sender_id")
    if not isinstance(sender_id, dict):
        return None
    for key in ("union_id", "open_id", "user_id"):
        value = sender_id.get(key)
        if value:
            return str(value)
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
    content = await _build_feishu_card_content(markdown_text, token)
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
    content = await _build_feishu_card_content(markdown_text, token)
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
    content = await _build_feishu_card_content(markdown_text, token)
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


async def _build_feishu_card_content(markdown_text: str, token: str) -> str:
    markdown_text, reference_sources = _rewrite_reference_links(markdown_text)
    elements = await _build_feishu_card_elements(markdown_text, token)
    if reference_sources:
        elements.append(_build_reference_sources_panel(reference_sources))
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


def _rewrite_reference_links(markdown: str) -> tuple[str, list[_ReferenceSource]]:
    markdown = _strip_model_generated_source_section(markdown)
    references_by_url: dict[str, _ReferenceSource] = {}
    in_code_block = False

    def replace_match(match: re.Match[str]) -> str:
        nonlocal references_by_url
        if in_code_block:
            return match.group(0)

        label = match.group(1).strip()
        url, existing_title = _parse_markdown_link_destination(match.group(2))
        reference_match = _reference_link_label_pattern.match(label)
        if not reference_match and existing_title:
            reference_match = _reference_link_label_pattern.match(existing_title)
        if not reference_match:
            return match.group(0)

        normalized_url = _normalize_reference_url(url)
        if normalized_url not in references_by_url:
            label_text = reference_match.group(1).replace(" ", "")
            description = existing_title or label
            references_by_url[normalized_url] = _ReferenceSource(
                number=len(references_by_url) + 1,
                label=label_text,
                description=description,
                url=url,
            )

        source = references_by_url[normalized_url]
        return f"[[{source.number}]]({_format_markdown_link_url(source.url)})"

    rewritten_lines: list[str] = []
    for line in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            rewritten_lines.append(line)
            continue
        rewritten_line = _markdown_link_pattern.sub(replace_match, line)
        rewritten_lines.append(_dedupe_adjacent_reference_links(rewritten_line))

    return "\n".join(rewritten_lines).strip(), list(references_by_url.values())


def _dedupe_adjacent_reference_links(markdown: str) -> str:
    previous = None
    current = markdown
    while current != previous:
        previous = current
        current = _adjacent_reference_pattern.sub(r"\g<ref>", current)
    return current


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
    return _resolve_lark_document_url(url) or _build_document_download_url(url)


def _is_lark_document_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    return host.endswith("feishu.cn") or host.endswith("larksuite.com")


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
    normalized_path = _normalize_reference_url(raw_path)
    query = urlencode({"path": normalized_path}, quote_via=quote)
    return f"{base_url.rstrip('/')}/api/v1/documents/download?{query}"


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
