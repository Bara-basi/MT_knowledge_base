from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from app.api.v1.query import ask_knowledge_base
from app.core.config import settings
from app.schemas.query import QueryRequest
from app.services.chat_records import create_chat_record, record_chat_answer
from app.services.evaluate.evaluate import evaluate_answer_fallback


router = APIRouter(prefix="/feishu", tags=["feishu"])
logger = logging.getLogger(__name__)

_tenant_access_token: str | None = None
_tenant_access_token_expires_at = 0.0
_seen_message_keys: dict[str, float] = {}
_seen_message_ttl_seconds = 3600
_pseudo_tag_pattern = re.compile(r"<(img|link)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)


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

        try:
            fallback_result = await asyncio.to_thread(
                evaluate_answer_fallback,
                question=question,
                answer=response.answer,
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
        if message_id:
            await _try_reply_feishu_markdown(
                message_id,
                "知识库暂时无法返回答案，请稍后再试。",
            )
        return

    if not message_id:
        _debug(
            "reply skipped",
            reason="missing_message_id",
            event_id=event_id,
            dedupe_key=dedupe_key,
        )
        return

    sent = await _try_reply_feishu_markdown(message_id, response.answer)
    _debug(
        "reply finished",
        event_id=event_id,
        message_id=message_id,
        dedupe_key=dedupe_key,
        sent=sent,
    )


async def _try_reply_feishu_markdown(message_id: str, markdown_text: str) -> bool:
    _debug(
        "markdown reply attempt",
        message_id=message_id,
        markdown_len=len(markdown_text),
        markdown_preview=_preview(markdown_text),
    )
    try:
        return await _reply_feishu_markdown(message_id, markdown_text)
    except HTTPException as exc:
        detail = _detail_to_text(exc.detail)
        logger.exception("Failed to send Feishu markdown card %s: %s", message_id, detail)
        _debug("markdown reply failed", message_id=message_id, error=detail)
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


async def _reply_feishu_markdown(message_id: str, markdown_text: str) -> bool:
    if not settings.feishu_app_id or not settings.feishu_app_secret:
        _debug(
            "markdown reply skipped",
            reason="missing_app_credentials",
            has_app_id=bool(settings.feishu_app_id),
            has_app_secret=bool(settings.feishu_app_secret),
        )
        return False

    token = await _get_tenant_access_token()
    timeout = httpx.Timeout(settings.feishu_timeout)
    url = f"{settings.feishu_base_url}/open-apis/im/v1/messages/{message_id}/reply"
    content = await _build_feishu_card_content(markdown_text, token)
    _debug(
        "markdown reply api request",
        message_id=message_id,
        url=url,
        content_preview=_preview(content),
    )

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

    _debug("markdown reply api success", message_id=message_id)
    return True


async def _build_feishu_card_content(markdown_text: str, token: str) -> str:
    elements = await _build_feishu_card_elements(markdown_text, token)
    if not elements:
        elements = [_build_markdown_element(" ")]

    return json.dumps(
        {
            "config": {
                "wide_screen_mode": True,
            },
            "elements": elements,
        },
        ensure_ascii=False,
    )


async def _build_feishu_card_elements(markdown_text: str, token: str) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    markdown_buffer: list[str] = []
    cursor = 0

    for match in _pseudo_tag_pattern.finditer(markdown_text):
        markdown_buffer.append(markdown_text[cursor : match.start()])
        tag = match.group(1).lower()
        value = match.group(2).strip()

        if tag == "link":
            markdown_buffer.append(_link_tag_to_markdown(value))
        elif tag == "img":
            _flush_markdown_buffer(elements, markdown_buffer)
            image_key = await _upload_local_image(value, token)
            if image_key:
                elements.append(
                    {
                        "tag": "img",
                        "img_key": image_key,
                        "alt": {
                            "tag": "plain_text",
                            "content": Path(value.replace("\\", "/")).name or "image",
                        },
                    }
                )

        cursor = match.end()

    markdown_buffer.append(markdown_text[cursor:])
    _flush_markdown_buffer(elements, markdown_buffer)
    return elements


def _flush_markdown_buffer(elements: list[dict[str, Any]], buffer: list[str]) -> None:
    markdown = "".join(buffer).strip()
    buffer.clear()
    if markdown:
        elements.append(_build_markdown_element(markdown))


def _build_markdown_element(markdown: str) -> dict[str, str]:
    return {
        "tag": "markdown",
        "content": _normalize_card_markdown(markdown),
    }


def _normalize_card_markdown(markdown: str) -> str:
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized_lines: list[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            normalized_lines.append(line)
            continue

        if in_code_block:
            normalized_lines.append(line)
            continue

        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            normalized_lines.extend(_heading_to_card_markdown(level, title))
            continue

        normalized_lines.append(_normalize_card_markdown_line(line))

    return "\n".join(normalized_lines).strip() or " "


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

