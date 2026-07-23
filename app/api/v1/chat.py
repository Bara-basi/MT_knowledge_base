from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.chat import (
    ConversationTopicContextRequest,
    ConversationTopicContextResponse,
    CreateConversationTopicRequest,
    CreateConversationTopicResponse,
    RecentConversationTopicsRequest,
    RecentConversationTopicsResponse,
    UpdateConversationTopicSummaryRequest,
    UpdateConversationTopicSummaryResponse,
)
from app.services.chat import (
    create_conversation_topic_record,
    get_conversation_topic_context,
    list_recent_conversation_topic_records,
    update_conversation_topic_summary,
)
from app.services.chat.topic_selection import remember_topic_selection


router = APIRouter(prefix="/chat", tags=["chat"])
NO_RECENT_CONVERSATION = "无近期对话"


@router.post(
    "/topics",
    response_model=CreateConversationTopicResponse,
    response_model_exclude_none=True,
)
async def create_topic(request: CreateConversationTopicRequest) -> CreateConversationTopicResponse:
    topic = await create_conversation_topic_record(
        user_id=request.user_id,
        session_id=request.session_id,
        topic=request.topic,
        summary=request.summary,
        metadata=request.metadata,
    )
    remember_topic_selection(
        user_id=request.user_id,
        session_id=request.session_id,
        topic_id=topic["topic_id"],
    )
    return CreateConversationTopicResponse(
        topic=_with_response_summary_fallback(topic),
        messages=[],
    )


@router.post(
    "/topics/recent",
    response_model=RecentConversationTopicsResponse,
    response_model_exclude_none=True,
)
async def list_recent_topics(
    request: RecentConversationTopicsRequest,
) -> RecentConversationTopicsResponse:
    topics = await list_recent_conversation_topic_records(
        user_id=request.user_id,
        session_id=request.session_id,
        limit=request.k,
    )
    return RecentConversationTopicsResponse(
        topics=topics,
        count=len(topics),
    )


@router.post(
    "/topics/context",
    response_model=ConversationTopicContextResponse,
    response_model_exclude_none=True,
)
async def get_topic_context(
    request: ConversationTopicContextRequest,
) -> ConversationTopicContextResponse:
    context = await get_conversation_topic_context(
        topic_id=request.topic_id,
        user_id=request.user_id,
        session_id=request.session_id,
        message_limit=request.k,
    )
    if not context:
        raise HTTPException(status_code=404, detail="conversation topic not found")
    updated_topic = await update_conversation_topic_summary(
        topic_id=request.topic_id,
        user_id=request.user_id,
        session_id=request.session_id,
        summary=request.summary,
    )
    if not updated_topic:
        raise HTTPException(status_code=404, detail="conversation topic not found")
    remember_topic_selection(
        user_id=request.user_id,
        session_id=request.session_id,
        topic_id=request.topic_id,
    )
    return ConversationTopicContextResponse(
        topic=updated_topic["topic"],
        messages=context["messages"],
    )


@router.post(
    "/topics/summary",
    response_model=UpdateConversationTopicSummaryResponse,
    response_model_exclude_none=True,
)
async def update_topic_summary(
    request: UpdateConversationTopicSummaryRequest,
) -> UpdateConversationTopicSummaryResponse:
    result = await update_conversation_topic_summary(
        topic_id=request.topic_id,
        user_id=request.user_id,
        session_id=request.session_id,
        summary=request.summary,
        topic=request.topic,
        conversation_id=request.conversation_id,
        message_increment=request.message_increment,
        metadata=request.metadata,
    )
    if not result:
        raise HTTPException(status_code=404, detail="conversation topic not found")
    return UpdateConversationTopicSummaryResponse(
        topic=result["topic"],
        bound_message=result["bound_message"] or None,
    )


def _with_response_summary_fallback(topic: dict) -> dict:
    response_topic = dict(topic)
    if not str(response_topic.get("summary") or "").strip():
        response_topic["summary"] = NO_RECENT_CONVERSATION
    return response_topic
