from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from psycopg import sql

from app.api.v1 import feishu
from app.core.config import settings
from app.db.postgres import ensure_chat_messages_table, postgres_connection


@dataclass(frozen=True)
class DailyUserChatStat:
    user_name: str
    message_count: int


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
) -> list[DailyUserChatStat]:
    ensure_chat_messages_table(table_name)
    report_date = target_date or default_report_date()
    start, end = day_bounds(target_date=report_date, timezone=timezone)
    table = sql.Identifier(table_name or settings.postgres_chat_table)

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
                    GROUP BY COALESCE(NULLIF(BTRIM(user_name), ''), '未识别用户')
                    ORDER BY message_count DESC, user_name ASC
                    """
                ).format(table=table),
                (start, end),
            )
            rows = cur.fetchall()

    return [
        DailyUserChatStat(
            user_name=str(row["user_name"]),
            message_count=int(row["message_count"]),
        )
        for row in rows
    ]


def build_daily_report_text(
    stats: list[DailyUserChatStat],
    *,
    target_date: date | None = None,
    generated_at: datetime | None = None,
) -> str:
    tz = get_report_timezone()
    generated = generated_at.astimezone(tz) if generated_at else datetime.now(tz)
    report_date = target_date or default_report_date(generated)
    total = sum(item.message_count for item in stats)

    lines = [
        f"MTSCO知识库昨日使用反馈（{report_date:%Y-%m-%d}）",
        "",
        f"截至今日 {generated:%H:%M}，昨日飞书问答入口共收到 {total} 条提问。",
    ]
    if stats:
        lines.append("昨日用户使用次数如下：")
        lines.extend(
            f"{index}. {item.user_name}：{item.message_count} 次"
            for index, item in enumerate(stats, start=1)
        )
    else:
        lines.append("昨日还没有产生新的问答记录。")
    return "\n".join(lines)


async def send_daily_report(
    *,
    target_union_id: str | None = None,
    target_open_id: str | None = None,
    target_session_id: str | None = None,
    target_date: date | None = None,
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
    stats = await asyncio.to_thread(fetch_daily_user_chat_stats, target_date=report_date)
    text = build_daily_report_text(stats, target_date=report_date)
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
        "stats": stats,
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


async def run_daily_report_loop(*, hour: int = 9, minute: int = 0) -> None:
    while True:
        await asyncio.sleep(seconds_until_next_daily_run(hour=hour, minute=minute))
        await send_daily_report()
