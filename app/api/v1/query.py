from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter

from app.schemas.query import QueryRequest, QueryResponse
from app.services.harness import ask_harness

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    response_model=QueryResponse,
    response_model_exclude_none=True,
)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    return await ask_knowledge_base(request)


async def ask_knowledge_base(
    request: QueryRequest,
    *,
    on_progress: Callable[[Any], None] | None = None,
) -> QueryResponse:
    """Run the knowledge-base Harness agent with no legacy fallback."""

    answer, _internal_session_id = await ask_harness(
        question=request.question,
        user_id=request.user_id,
        source_session_id=request.session_id,
        on_progress=on_progress,
    )
    return QueryResponse(question=request.question, answer=answer, topic_id=None)
