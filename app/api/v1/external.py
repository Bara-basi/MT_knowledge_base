from __future__ import annotations

import asyncio
import logging
from pathlib import PurePosixPath
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.api.v1.query import ask_knowledge_base
from app.schemas.external import (
    ExternalQueryRequest,
    ExternalQueryResponse,
    QuoteScoreResponse,
)
from app.schemas.query import QueryRequest, QueryResponse
from app.services.external_answer_formatting import format_external_markdown_answer
from app.services.external_chat_records import (
    canonical_external_ids,
    create_external_chat_record,
    record_external_chat_answer,
)
from app.services.quote_scoring import (
    JSON_OUTPUT_PROMPT,
    QUOTE_SCORING_PROMPT,
    parse_json_answer,
    parse_quote_score,
    spreadsheet_to_compact_json,
)


router = APIRouter(prefix="/external", tags=["external"])
logger = logging.getLogger(__name__)


@router.post(
    "/query",
    response_model=ExternalQueryResponse,
    response_model_exclude_none=True,
)
async def query_external_knowledge_base(
    request: ExternalQueryRequest,
) -> ExternalQueryResponse:
    """Run the standard QA workflow with isolated external history storage."""

    response = await _execute_external_query(
        request,
        additional_system_prompt=(
            JSON_OUTPUT_PROMPT if request.format_type == "json" else ""
        ),
    )
    if request.format_type == "json":
        try:
            answer: str | dict[str, Any] | list[Any] = parse_json_answer(response.answer)
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"n8n returned invalid JSON: {exc}",
            ) from exc
    else:
        answer = format_external_markdown_answer(
            response.answer,
            use_lark_document=request.use_lark_document,
        )

    return ExternalQueryResponse(
        question=request.question,
        answer=answer,
        user_id=request.user_id,
        service_id=request.service_id,
        session_id=request.session_id,
        topic_id=response.topic_id,
        answer_format=request.format_type,
    )


@router.post(
    "/quote-score",
    response_model=QuoteScoreResponse,
    response_model_exclude_none=True,
)
async def score_external_quote(
    question: str = Form(...),
    user_id: str = Form(...),
    service_id: str = Form(...),
    session_id: str = Form(...),
    use_lark_document: bool = Form(False),
    file: UploadFile | None = File(None),
) -> QuoteScoreResponse:
    """Score quote text or an uploaded Excel workbook through the QA workflow."""

    try:
        request = ExternalQueryRequest(
            question=question,
            user_id=user_id,
            service_id=service_id,
            session_id=session_id,
            use_lark_document=use_lark_document,
            format_type="json",
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_context=False),
        ) from exc
    file_name: str | None = None
    if file is not None:
        file_name = PurePosixPath(
            str(file.filename or "quote.xlsx").replace("\\", "/")
        ).name[:255]
        try:
            content = await file.read()
            task_input = await asyncio.to_thread(
                spreadsheet_to_compact_json,
                filename=file_name,
                content=content,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await file.close()
    else:
        task_input = _plain_quote_input(request.question)

    response = await _execute_external_query(
        request,
        additional_system_prompt=QUOTE_SCORING_PROMPT,
        task_input=task_input,
        tool_name="quote_scoring",
    )
    try:
        result = parse_quote_score(response.answer)
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"n8n returned an invalid quote score: {exc}",
        ) from exc

    return QuoteScoreResponse(
        **result.model_dump(),
        user_id=request.user_id,
        service_id=request.service_id,
        session_id=request.session_id,
        file_name=file_name,
        topic_id=response.topic_id,
    )


async def _execute_external_query(
    request: ExternalQueryRequest,
    *,
    additional_system_prompt: str = "",
    task_input: str = "",
    tool_name: str | None = None,
) -> QueryResponse:
    """Persist and execute one isolated external query for all external tools."""

    canonical_ids = canonical_external_ids(
        service_id=request.service_id,
        user_id=request.user_id,
        session_id=request.session_id,
    )
    try:
        pending_row = await create_external_chat_record(
            service_id=request.service_id,
            user_id=canonical_ids["user_id"],
            session_id=canonical_ids["session_id"],
            question=request.question,
        )
    except Exception as exc:  # noqa: BLE001 - database failures need a stable API error.
        logger.exception("Failed to create external chat record: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="external chat storage is unavailable",
        ) from exc

    response = await ask_knowledge_base(
        QueryRequest(
            question=request.question,
            user_id=canonical_ids["user_id"],
            session_id=canonical_ids["session_id"],
            conversation_id=canonical_ids["session_id"],
            metadata={
                "source": "external",
                "service_id": request.service_id,
                "format_type": request.format_type,
                **({"tool": tool_name} if tool_name else {}),
            },
            service_id=request.service_id,
            use_lark_document=request.use_lark_document,
            format_type=request.format_type,
            additional_system_prompt=additional_system_prompt,
            task_input=task_input,
            source="external",
        )
    )

    try:
        await record_external_chat_answer(
            message_id=pending_row["message_id"],
            service_id=request.service_id,
            user_id=canonical_ids["user_id"],
            session_id=canonical_ids["session_id"],
            answer=response.answer,
            topic_id=response.topic_id,
        )
    except Exception as exc:  # noqa: BLE001 - do not report success without isolated persistence.
        logger.exception("Failed to complete external chat record: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="external chat answer could not be persisted",
        ) from exc

    return response


def _plain_quote_input(question: str) -> str:
    import json

    return json.dumps(
        {"input_type": "text", "content": question},
        ensure_ascii=False,
        separators=(",", ":"),
    )
