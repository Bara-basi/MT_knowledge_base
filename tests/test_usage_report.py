from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.services.usage_report import (
    UsageReportData,
    UsageUserStat,
    build_usage_report_data,
    build_usage_report_text,
    normalize_department_names,
)
from app.services import usage_report


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _row(
    user_id: str,
    user_name: str,
    create_time: datetime,
    *,
    answered: bool = True,
) -> dict:
    return {
        "user_id": user_id,
        "user_name": user_name,
        "create_time": create_time,
        "is_answered": answered,
    }


def test_daily_usage_data_calculates_multiple_dimensions() -> None:
    start = datetime(2026, 7, 21, 0, 0, tzinfo=SHANGHAI)
    end = datetime(2026, 7, 22, 0, 0, tzinfo=SHANGHAI)
    data = build_usage_report_data(
        current_rows=[
            _row("u1", "Alice", datetime(2026, 7, 21, 9, 5, tzinfo=SHANGHAI)),
            _row("u1", "Alice", datetime(2026, 7, 21, 9, 20, tzinfo=SHANGHAI)),
            _row(
                "u2",
                "Bob",
                datetime(2026, 7, 21, 15, 0, tzinfo=SHANGHAI),
                answered=False,
            ),
        ],
        previous_rows=[
            _row("u1", "Alice", datetime(2026, 7, 20, 10, 0, tzinfo=SHANGHAI))
        ],
        users_seen_before={"u1"},
        start=start,
        end=end,
        granularity="daily",
        timezone=SHANGHAI,
    )

    assert data.total_questions == 3
    assert data.active_users == 2
    assert data.new_users == 1
    assert data.answered_questions == 2
    assert data.previous_total_questions == 1
    assert data.previous_active_users == 1
    assert data.questions_per_user == 1.5
    assert data.answer_completion_rate == 2 / 3 * 100
    assert data.peak_label == "09:00–10:00"
    assert data.peak_count == 2
    assert sum(item.message_count for item in data.buckets) == 3
    assert data.user_stats[0] == UsageUserStat("Alice", 2, 2)


def test_weekly_trend_uses_seven_equal_1730_to_1730_buckets() -> None:
    start = datetime(2026, 7, 10, 17, 30, tzinfo=SHANGHAI)
    end = datetime(2026, 7, 17, 17, 30, tzinfo=SHANGHAI)
    data = build_usage_report_data(
        current_rows=[
            _row("u1", "Alice", datetime(2026, 7, 10, 18, 0, tzinfo=SHANGHAI)),
            _row("u1", "Alice", datetime(2026, 7, 17, 16, 0, tzinfo=SHANGHAI)),
        ],
        previous_rows=[],
        users_seen_before=set(),
        start=start,
        end=end,
        granularity="weekly",
        timezone=SHANGHAI,
    )

    assert len(data.buckets) == 7
    assert data.buckets[0].label == "周六 07-11"
    assert data.buckets[-1].label == "周五 07-17"
    assert data.buckets[0].message_count == 1
    assert data.buckets[-1].message_count == 1
    assert sum(item.message_count for item in data.buckets) == data.total_questions


def test_usage_report_text_has_consistent_visual_sections() -> None:
    text = build_usage_report_text(
        UsageReportData(
            user_stats=[UsageUserStat("Alice", 6, 6), UsageUserStat("Bob", 4, 3)],
            total_questions=10,
            active_users=2,
            new_users=1,
            answered_questions=9,
            previous_total_questions=8,
            previous_active_users=2,
            peak_label="14:00–15:00",
            peak_count=4,
        ),
        title="MTSCO 知识库 · 日使用简报",
        period_text="2026-07-21（昨日）",
        comparison_label="前一日",
        distribution_title="时段分布",
    )

    assert "💬 **10** 条提问" in text
    assert "👥 **2** 位活跃用户" in text
    assert "✅ 回答完成率 **90.0%**" in text
    assert "提问量 ↑ 25.0%" in text
    assert "🥇 Alice" in text
    assert "██████" in text
    assert "统计范围：全部用户" in text


def test_normalize_department_names_accepts_postgres_array_display() -> None:
    assert normalize_department_names("{迈拓思学园}") == ("迈拓思学园",)
    assert normalize_department_names(["迈拓思学园,AI部", "AI部"]) == (
        "迈拓思学园",
        "AI部",
    )


def test_fetch_usage_report_data_filters_all_comparison_scopes(monkeypatch) -> None:
    calls: list[tuple[object, tuple]] = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, query, params):
            calls.append((query, params))

        def fetchall(self):
            return []

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

    class FakeConnectionContext:
        def __enter__(self):
            return FakeConnection()

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(usage_report, "ensure_chat_messages_table", lambda _name=None: None)
    monkeypatch.setattr(usage_report, "postgres_connection", lambda: FakeConnectionContext())
    start = datetime(2026, 7, 21, 0, 0, tzinfo=SHANGHAI)
    end = datetime(2026, 7, 22, 0, 0, tzinfo=SHANGHAI)

    data = usage_report.fetch_usage_report_data(
        start=start,
        end=end,
        granularity="daily",
        timezone=SHANGHAI,
        department_names="{迈拓思学园}",
    )

    assert calls[0][1] == (datetime(2026, 7, 20, 0, 0, tzinfo=SHANGHAI), end, ["迈拓思学园"])
    assert calls[1][1] == (start, ["迈拓思学园"])
    assert data.department_names == ("迈拓思学园",)
    assert "department_names" in str(calls[0][0])
