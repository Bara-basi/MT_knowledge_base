from __future__ import annotations

import asyncio
from datetime import datetime, time, timedelta
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


FRIDAY_WEEKDAY = 4
REPORT_CUTOFF = time(hour=17, minute=30)


WeeklyUserChatStat = UsageUserStat


def get_report_timezone() -> ZoneInfo:
    try:
        return ZoneInfo(settings.weekly_report_timezone or "Asia/Shanghai")
    except ZoneInfoNotFoundError:
        return ZoneInfo("Asia/Shanghai")


def default_report_end(now: datetime | None = None) -> datetime:
    """Return the most recent completed Friday 17:30 cutoff."""
    tz = get_report_timezone()
    current = now.astimezone(tz) if now else datetime.now(tz)
    days_since_friday = (current.weekday() - FRIDAY_WEEKDAY) % 7
    friday = current.date() - timedelta(days=days_since_friday)
    report_end = datetime.combine(friday, REPORT_CUTOFF, tzinfo=tz)
    if report_end > current:
        report_end -= timedelta(days=7)
    return report_end


def week_bounds(
    report_end: datetime | None = None,
    timezone: ZoneInfo | None = None,
) -> tuple[datetime, datetime]:
    tz = timezone or get_report_timezone()
    if report_end is None:
        end = default_report_end()
    elif report_end.tzinfo is None:
        end = report_end.replace(tzinfo=tz)
    else:
        end = report_end.astimezone(tz)
    return end - timedelta(days=7), end


def fetch_weekly_user_chat_stats(
    *,
    report_end: datetime | None = None,
    table_name: str | None = None,
    timezone: ZoneInfo | None = None,
    department_names: Iterable[str] | str | None = None,
) -> list[WeeklyUserChatStat]:
    ensure_chat_messages_table(table_name)
    start, end = week_bounds(report_end=report_end, timezone=timezone)
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
        WeeklyUserChatStat(
            user_name=str(row["user_name"]),
            message_count=int(row["message_count"]),
        )
        for row in rows
    ]


def fetch_weekly_report_data(
    *,
    report_end: datetime | None = None,
    table_name: str | None = None,
    timezone: ZoneInfo | None = None,
    department_names: Iterable[str] | str | None = None,
) -> UsageReportData:
    tz = timezone or get_report_timezone()
    start, end = week_bounds(report_end=report_end, timezone=tz)
    return fetch_usage_report_data(
        start=start,
        end=end,
        granularity="weekly",
        timezone=tz,
        table_name=table_name,
        department_names=department_names,
    )


def build_weekly_report_text(
    stats: list[WeeklyUserChatStat],
    *,
    report_end: datetime | None = None,
    report_data: UsageReportData | None = None,
) -> str:
    start, end = week_bounds(report_end=report_end)
    data = report_data or _basic_report_data(stats)
    return build_usage_report_text(
        data,
        title="MTSCO 知识库 · 周使用简报",
        period_text=f"{start:%Y-%m-%d %H:%M} — {end:%Y-%m-%d %H:%M}",
        comparison_label="上一周期",
        distribution_title="每日趋势（按 17:30 截止）",
    )


def _basic_report_data(stats: list[WeeklyUserChatStat]) -> UsageReportData:
    total = sum(item.message_count for item in stats)
    return UsageReportData(
        user_stats=list(stats),
        total_questions=total,
        active_users=len(stats),
        answered_questions=total,
    )


async def send_weekly_report(
    *,
    target_union_id: str | None = None,
    target_open_id: str | None = None,
    target_session_id: str | None = None,
    report_end: datetime | None = None,
    department_names: Iterable[str] | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if not force and not settings.weekly_report_enabled:
        return {
            "ok": False,
            "skipped": True,
            "reason": "weekly_report_disabled",
            "message_id": None,
        }

    session_id = (target_session_id or settings.weekly_report_target_session_id).strip()
    union_id = (
        target_union_id or target_open_id or settings.weekly_report_target_union_id
    ).strip()
    if not union_id and not session_id:
        raise ValueError("Weekly report target union_id or session_id is required.")

    end = report_end or default_report_end()
    start, end = week_bounds(report_end=end)
    departments = normalize_department_names(
        department_names
        if department_names is not None
        else getattr(settings, "weekly_report_departments", "")
    )
    fetch_kwargs: dict[str, Any] = {"report_end": end}
    if departments:
        fetch_kwargs["department_names"] = departments
    report_data = await asyncio.to_thread(fetch_weekly_report_data, **fetch_kwargs)
    stats = report_data.user_stats
    text = build_weekly_report_text(stats, report_end=end, report_data=report_data)
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
        "report_start": start,
        "report_end": end,
        "department_names": departments,
        "stats": stats,
        "report_data": report_data,
        "text": text,
    }


def seconds_until_next_weekly_run(
    *,
    now: datetime | None = None,
    weekday: int = FRIDAY_WEEKDAY,
    hour: int = 17,
    minute: int = 35,
    timezone: ZoneInfo | None = None,
) -> float:
    tz = timezone or get_report_timezone()
    current = now.astimezone(tz) if now else datetime.now(tz)
    days_ahead = (weekday - current.weekday()) % 7
    next_date = current.date() + timedelta(days=days_ahead)
    next_run = datetime.combine(next_date, time(hour=hour, minute=minute), tzinfo=tz)
    if next_run <= current:
        next_run += timedelta(days=7)
    return max((next_run - current).total_seconds(), 0.0)


async def run_weekly_report_loop(
    *,
    weekday: int = FRIDAY_WEEKDAY,
    hour: int = 17,
    minute: int = 35,
    department_names: Iterable[str] | str | None = None,
) -> None:
    while True:
        await asyncio.sleep(
            seconds_until_next_weekly_run(weekday=weekday, hour=hour, minute=minute)
        )
        await send_weekly_report(department_names=department_names)
