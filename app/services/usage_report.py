from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Literal
from zoneinfo import ZoneInfo

from psycopg import sql

from app.core.config import settings
from app.db.postgres import ensure_chat_messages_table, postgres_connection


ReportGranularity = Literal["daily", "weekly"]
WEEKDAY_LABELS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


@dataclass(frozen=True)
class UsageUserStat:
    user_name: str
    message_count: int
    answered_count: int = 0


@dataclass(frozen=True)
class UsageBucketStat:
    label: str
    message_count: int


@dataclass(frozen=True)
class UsageReportData:
    user_stats: list[UsageUserStat] = field(default_factory=list)
    total_questions: int = 0
    active_users: int = 0
    new_users: int = 0
    answered_questions: int = 0
    previous_total_questions: int = 0
    previous_active_users: int = 0
    buckets: list[UsageBucketStat] = field(default_factory=list)
    peak_label: str | None = None
    peak_count: int = 0
    department_names: tuple[str, ...] = ()

    @property
    def questions_per_user(self) -> float:
        return self.total_questions / self.active_users if self.active_users else 0.0

    @property
    def answer_completion_rate(self) -> float:
        if not self.total_questions:
            return 0.0
        return self.answered_questions / self.total_questions * 100

    @property
    def top_three_share(self) -> float:
        if not self.total_questions:
            return 0.0
        top_three = sum(item.message_count for item in self.user_stats[:3])
        return top_three / self.total_questions * 100


def fetch_usage_report_data(
    *,
    start: datetime,
    end: datetime,
    granularity: ReportGranularity,
    timezone: ZoneInfo,
    table_name: str | None = None,
    department_names: Iterable[str] | str | None = None,
) -> UsageReportData:
    if end <= start:
        raise ValueError("Usage report end must be later than start.")

    ensure_chat_messages_table(table_name)
    table = sql.Identifier(table_name or settings.postgres_chat_table)
    previous_start = start - (end - start)
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
                        user_id,
                        COALESCE(NULLIF(BTRIM(user_name), ''), '未识别用户') AS user_name,
                        create_time,
                        (NULLIF(answer, '') IS NOT NULL) AS is_answered
                    FROM {table}
                    WHERE create_time >= %s
                      AND create_time < %s
                      {department_condition}
                    ORDER BY create_time ASC
                    """
                ).format(table=table, department_condition=department_condition),
                (previous_start, end, list(departments)) if departments else (previous_start, end),
            )
            rows = [dict(row) for row in cur.fetchall()]
            cur.execute(
                sql.SQL(
                    """
                    SELECT DISTINCT user_id
                    FROM {table}
                    WHERE create_time < %s
                      {department_condition}
                    """
                ).format(table=table, department_condition=department_condition),
                (start, list(departments)) if departments else (start,),
            )
            users_seen_before = {str(row["user_id"]) for row in cur.fetchall()}

    current_rows = [row for row in rows if start <= row["create_time"] < end]
    previous_rows = [row for row in rows if previous_start <= row["create_time"] < start]
    return build_usage_report_data(
        current_rows=current_rows,
        previous_rows=previous_rows,
        users_seen_before=users_seen_before,
        start=start,
        end=end,
        granularity=granularity,
        timezone=timezone,
        department_names=departments,
    )


def build_usage_report_data(
    *,
    current_rows: list[dict],
    previous_rows: list[dict],
    users_seen_before: set[str],
    start: datetime,
    end: datetime,
    granularity: ReportGranularity,
    timezone: ZoneInfo,
    department_names: Iterable[str] | str | None = None,
) -> UsageReportData:
    user_counts: Counter[str] = Counter()
    user_answered_counts: Counter[str] = Counter()
    latest_user_names: dict[str, str] = {}
    local_times: list[datetime] = []

    for row in current_rows:
        user_id = str(row["user_id"])
        user_counts[user_id] += 1
        if row.get("is_answered"):
            user_answered_counts[user_id] += 1
        latest_user_names[user_id] = str(row.get("user_name") or "未识别用户")
        local_times.append(row["create_time"].astimezone(timezone))

    user_stats = sorted(
        (
            UsageUserStat(
                user_name=latest_user_names[user_id],
                message_count=message_count,
                answered_count=user_answered_counts[user_id],
            )
            for user_id, message_count in user_counts.items()
        ),
        key=lambda item: (-item.message_count, item.user_name),
    )
    current_user_ids = set(user_counts)
    previous_user_ids = {str(row["user_id"]) for row in previous_rows}
    buckets, peak_label, peak_count = _build_time_distribution(
        local_times=local_times,
        start=start.astimezone(timezone),
        end=end.astimezone(timezone),
        granularity=granularity,
    )
    return UsageReportData(
        user_stats=user_stats,
        total_questions=len(current_rows),
        active_users=len(current_user_ids),
        new_users=len(current_user_ids - users_seen_before),
        answered_questions=sum(1 for row in current_rows if row.get("is_answered")),
        previous_total_questions=len(previous_rows),
        previous_active_users=len(previous_user_ids),
        buckets=buckets,
        peak_label=peak_label,
        peak_count=peak_count,
        department_names=normalize_department_names(department_names),
    )


def normalize_department_names(values: Iterable[str] | str | None) -> tuple[str, ...]:
    """Normalize CLI/env input, including PostgreSQL's ``{department}`` display."""

    if values is None:
        return ()
    raw_values = [values] if isinstance(values, str) else list(values)
    normalized: list[str] = []
    for raw_value in raw_values:
        text = str(raw_value or "").strip()
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        for item in text.split(","):
            name = item.strip().strip('"').strip()
            if name and name not in normalized:
                normalized.append(name)
    return tuple(normalized)


def _build_time_distribution(
    *,
    local_times: list[datetime],
    start: datetime,
    end: datetime,
    granularity: ReportGranularity,
) -> tuple[list[UsageBucketStat], str | None, int]:
    if granularity == "daily":
        segment_counts = Counter(item.hour // 6 for item in local_times)
        buckets = [
            UsageBucketStat(label, segment_counts[index])
            for index, label in enumerate(("00–06", "06–12", "12–18", "18–24"))
        ]
        hourly_counts = Counter(item.hour for item in local_times)
        if not hourly_counts:
            return buckets, None, 0
        peak_hour, peak_count = min(
            hourly_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return buckets, f"{peak_hour:02d}:00–{(peak_hour + 1):02d}:00", peak_count

    day_count = max(round((end - start).total_seconds() / 86400), 1)
    bucket_counts: Counter[int] = Counter()
    for item in local_times:
        bucket_index = int((item - start).total_seconds() // 86400)
        if 0 <= bucket_index < day_count:
            bucket_counts[bucket_index] += 1
    buckets = [
        UsageBucketStat(
            f"{WEEKDAY_LABELS[bucket_end.weekday()]} {bucket_end:%m-%d}",
            bucket_counts[offset],
        )
        for offset in range(day_count)
        for bucket_end in [(start + timedelta(days=offset + 1)).date()]
    ]
    if not bucket_counts:
        return buckets, None, 0
    peak_index, peak_count = min(
        bucket_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    peak_date = (start + timedelta(days=peak_index + 1)).date()
    return (
        buckets,
        f"{WEEKDAY_LABELS[peak_date.weekday()]} {peak_date:%m-%d}",
        peak_count,
    )


def build_usage_report_text(
    data: UsageReportData,
    *,
    title: str,
    period_text: str,
    comparison_label: str,
    distribution_title: str,
) -> str:
    lines = [
        f"📊 **{title}**",
        f"<font color='grey'>{period_text}</font>",
        "",
        "💡 **核心数据**",
        f"💬 **{data.total_questions}** 条提问　　👥 **{data.active_users}** 位活跃用户",
        f"🆕 **{data.new_users}** 位首次使用　　⚡ **{data.questions_per_user:.1f}** 条/人",
        f"✅ 回答完成率 **{data.answer_completion_rate:.1f}%**",
        "",
        f"📈 **较{comparison_label}**",
        f"提问量 {_format_change(data.total_questions, data.previous_total_questions)}　　"
        f"活跃人数 {_format_change(data.active_users, data.previous_active_users)}",
        "",
        f"⏰ **{distribution_title}**",
    ]
    if data.peak_label:
        lines.append(f"高峰：**{data.peak_label}**（{data.peak_count} 条）")
    else:
        lines.append("本周期暂无活跃时段。")
    lines.extend(_format_buckets(data.buckets))
    lines.extend(["", "🏆 **用户活跃榜**"])
    if data.user_stats:
        lines.extend(_format_user_ranking(data.user_stats, data.total_questions))
        lines.append(f"Top 3 用户贡献占比：**{data.top_three_share:.1f}%**")
    else:
        lines.append("本周期还没有产生新的问答记录。")
    scope = "、".join(data.department_names) if data.department_names else "全部用户"
    lines.extend(
        [
            "",
            f"<font color='grey'>统计范围：{scope}</font>",
        ]
    )
    return "\n".join(lines)


def _format_change(current: int, previous: int) -> str:
    if previous == 0:
        return "→ 持平" if current == 0 else f"↑ 新增 {current}"
    rate = (current - previous) / previous * 100
    if rate > 0:
        return f"↑ {rate:.1f}%"
    if rate < 0:
        return f"↓ {abs(rate):.1f}%"
    return "→ 持平"


def _format_buckets(buckets: list[UsageBucketStat]) -> list[str]:
    if not buckets:
        return []
    maximum = max((item.message_count for item in buckets), default=0)
    return [
        f"{item.label}　{_bar(item.message_count, maximum)} {item.message_count}"
        for item in buckets
    ]


def _format_user_ranking(
    user_stats: list[UsageUserStat],
    total_questions: int,
    *,
    limit: int = 10,
) -> list[str]:
    medals = ("🥇", "🥈", "🥉")
    maximum = user_stats[0].message_count if user_stats else 0
    lines: list[str] = []
    for index, item in enumerate(user_stats[:limit], start=1):
        rank = medals[index - 1] if index <= len(medals) else f"{index}."
        share = item.message_count / total_questions * 100 if total_questions else 0.0
        lines.append(
            f"{rank} {item.user_name}　{_bar(item.message_count, maximum, width=6)} "
            f"**{item.message_count}** 条 · {share:.1f}%"
        )
    return lines


def _bar(value: int, maximum: int, *, width: int = 8) -> str:
    if maximum <= 0 or value <= 0:
        filled = 0
    else:
        filled = max(1, round(value / maximum * width))
    return "█" * filled + "░" * (width - filled)
