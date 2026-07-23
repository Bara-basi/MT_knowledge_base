from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta
from typing import Any, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from psycopg import sql

from app.api.v1 import feishu
from app.core.config import settings
from app.db.postgres import ensure_chat_messages_table, postgres_connection
from app.services.usage_report import (
    UsageReportData,
    UsageUserStat,
    build_usage_report_text,
    fetch_usage_report_data,
    normalize_department_names,
)


DailyUserChatStat = UsageUserStat


def get_report_timezone() -> ZoneInfo:
    try:
        return ZoneInfo(settings.daily_report_timezone or "Asia/Shanghai")
    except ZoneInfoNotFoundError:
        return ZoneInfo("Asia/Shanghai")


def default_report_date(now: datetime | None = None) -> date:
    tz = get_report_timezone()
    current = now.astimezone(tz) if now else datetime.now(tz)
    return current.date() - timedelta(days=1)


def day_bounds(
    target_date: date | None = None,
    timezone: ZoneInfo | None = None,
) -> tuple[datetime, datetime]:
    tz = timezone or get_report_timezone()
    day = target_date or default_report_date()
    start = datetime.combine(day, time.min, tzinfo=tz)
    return start, start + timedelta(days=1)


def fetch_daily_user_chat_stats(
    *,
    target_date: date | None = None,
    table_name: str | None = None,
    timezone: ZoneInfo | None = None,
    department_names: Iterable[str] | str | None = None,
) -> list[DailyUserChatStat]:
    ensure_chat_messages_table(table_name)
    report_date = target_date or default_report_date()
    start, end = day_bounds(target_date=report_date, timezone=timezone)
    table = sql.Identifier(table_name or settings.postgres_chat_table)
    departments = normalize_department_names(department_names)
    department_condition = (
        sql.SQL(" AND COALESCE(department_names, ARRAY[]::TEXT[]) && %s::TEXT[]")
        if departments
        else sql.SQL("")
    )

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT
                        COALESCE(NULLIF(BTRIM(user_name), ''), '未识别用户') AS user_name,
                        COUNT(*)::int AS message_count
                    FROM {table}
                    WHERE create_time >= %s
                      AND create_time < %s
                      {department_condition}
                    GROUP BY COALESCE(NULLIF(BTRIM(user_name), ''), '未识别用户')
                    ORDER BY message_count DESC, user_name ASC
                    """
                ).format(table=table, department_condition=department_condition),
                (start, end, list(departments)) if departments else (start, end),
            )
            rows = cur.fetchall()

    return [
        DailyUserChatStat(
            user_name=str(row["user_name"]),
            message_count=int(row["message_count"]),
        )
        for row in rows
    ]


def fetch_daily_report_data(
    *,
    target_date: date | None = None,
    table_name: str | None = None,
    timezone: ZoneInfo | None = None,
    department_names: Iterable[str] | str | None = None,
) -> UsageReportData:
    tz = timezone or get_report_timezone()
    start, end = day_bounds(target_date=target_date, timezone=tz)
    return fetch_usage_report_data(
        start=start,
        end=end,
        granularity="daily",
        timezone=tz,
        table_name=table_name,
        department_names=department_names,
    )


def build_daily_report_text(
    stats: list[DailyUserChatStat],
    *,
    target_date: date | None = None,
    generated_at: datetime | None = None,
    report_data: UsageReportData | None = None,
) -> str:
    tz = get_report_timezone()
    generated = generated_at.astimezone(tz) if generated_at else datetime.now(tz)
    report_date = target_date or default_report_date(generated)
    data = report_data or _basic_report_data(stats)
    return build_usage_report_text(
        data,
        title="MTSCO 知识库 · 日使用简报",
        period_text=f"{report_date:%Y-%m-%d}（昨日） · 生成于 {generated:%H:%M}",
        comparison_label="前一日",
        distribution_title="时段分布",
    )


def _basic_report_data(stats: list[DailyUserChatStat]) -> UsageReportData:
    total = sum(item.message_count for item in stats)
    return UsageReportData(
        user_stats=list(stats),
        total_questions=total,
        active_users=len(stats),
        answered_questions=total,
    )


async def send_daily_report(
    *,
    target_union_id: str | None = None,
    target_open_id: str | None = None,
    target_session_id: str | None = None,
    target_date: date | None = None,
    department_names: Iterable[str] | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if not force and not settings.daily_report_enabled:
        return {
            "ok": False,
            "skipped": True,
            "reason": "daily_report_disabled",
            "message_id": None,
        }

    session_id = (target_session_id or settings.daily_report_target_session_id).strip()
    union_id = (target_union_id or target_open_id or settings.daily_report_target_union_id).strip()
    if not union_id and not session_id:
        raise ValueError("Daily report target union_id or session_id is required.")

    report_date = target_date or default_report_date()
    departments = normalize_department_names(
        department_names
        if department_names is not None
        else getattr(settings, "daily_report_departments", "")
    )
    fetch_kwargs: dict[str, Any] = {"target_date": report_date}
    if departments:
        fetch_kwargs["department_names"] = departments
    report_data = await asyncio.to_thread(fetch_daily_report_data, **fetch_kwargs)
    stats = report_data.user_stats
    text = build_daily_report_text(
        stats,
        target_date=report_date,
        report_data=report_data,
    )
    receive_id_type = "union_id" if union_id else "chat_id"
    receive_id = union_id or session_id
    if union_id:
        message_id = await feishu._send_feishu_markdown_message_to_receive_id(
            receive_id=union_id,
            receive_id_type="union_id",
            markdown_text=text,
            log_content=True,
        )
    else:
        message_id = await feishu._try_send_feishu_markdown_message(
            chat_id=session_id,
            reply_to_message_id=None,
            markdown_text=text,
            log_content=True,
        )
    return {
        "ok": message_id is not None,
        "message_id": message_id,
        "target_union_id": union_id,
        "target_session_id": session_id,
        "receive_id": receive_id,
        "receive_id_type": receive_id_type,
        "report_date": report_date,
        "department_names": departments,
        "stats": stats,
        "report_data": report_data,
        "text": text,
    }


def seconds_until_next_daily_run(
    *,
    now: datetime | None = None,
    hour: int = 9,
    minute: int = 0,
    timezone: ZoneInfo | None = None,
) -> float:
    tz = timezone or get_report_timezone()
    current = now.astimezone(tz) if now else datetime.now(tz)
    next_run = datetime.combine(current.date(), time(hour=hour, minute=minute), tzinfo=tz)
    if current >= next_run:
        next_run += timedelta(days=1)
    return max((next_run - current).total_seconds(), 0.0)


async def run_daily_report_loop(
    *,
    hour: int = 9,
    minute: int = 0,
    department_names: Iterable[str] | str | None = None,
) -> None:
    while True:
        await asyncio.sleep(seconds_until_next_daily_run(hour=hour, minute=minute))
        await send_daily_report(department_names=department_names)
