from __future__ import annotations

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
    """Forward a user question to the n8n QA agent and normalize its answer."""
    payload = N8nQueryRequest(**request.model_dump()).model_dump()
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
    return text[:500] if text else "<empty response>"


def _missing_answer_detail(raw_response: Any) -> str:
    preview = str(raw_response)
    if len(preview) > 500:
        preview = f"{preview[:500]}..."
    return (
        "n8n query webhook response did not contain an answer. "
        "Expected one of: answer, output, text, message, result, data. "
        f"Response preview: {preview}"
    )
