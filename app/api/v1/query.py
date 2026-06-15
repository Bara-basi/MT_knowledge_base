from __future__ import annotations

import re
import unicodedata
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.query import N8nQueryRequest, QueryRequest, QueryResponse


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
    payload = N8nQueryRequest(**n8n_request.model_dump()).model_dump()
    print(
        "[query] n8n request "
        f"url={settings.n8n_query_webhook_url!r} "
        f"question={sanitized_question!r} "
        f"question_sanitized={sanitized_question != request.question!r} "
        f"user_id={request.user_id!r} "
        f"session_id={request.session_id!r} "
        f"conversation_id={request.conversation_id!r} "
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

    if not answer:
        raise HTTPException(
            status_code=502,
            detail=_missing_answer_detail(raw_response),
        )

    return QueryResponse(
        question=request.question,
        answer=answer,
    )


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
