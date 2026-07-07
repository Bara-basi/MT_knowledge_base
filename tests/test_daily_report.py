from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from app.services import daily_report


def test_build_daily_report_text_lists_user_counts() -> None:
    text = daily_report.build_daily_report_text(
        [
            daily_report.DailyUserChatStat("Alice", 3),
            daily_report.DailyUserChatStat("Bob", 1),
        ],
        target_date=date(2026, 7, 7),
        generated_at=datetime(2026, 7, 7, 9, 5, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert "MTSCO知识库每日使用反馈（2026-07-07）" in text
    assert "今日飞书问答入口共收到 4 条提问" in text
    assert "1. Alice：3 次" in text
    assert "2. Bob：1 次" in text
    assert "PostgreSQL" not in text
    assert "chat_messages" not in text


def test_fetch_daily_user_chat_stats_queries_current_day(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, query, params):
            captured["query"] = query
            captured["params"] = params

        def fetchall(self):
            return [
                {"user_name": "Alice", "message_count": 2},
                {"user_name": "未识别用户", "message_count": 1},
            ]

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

    class FakeConnectionContext:
        def __enter__(self):
            return FakeConnection()

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(daily_report, "ensure_chat_messages_table", lambda table_name=None: None)
    monkeypatch.setattr(daily_report, "postgres_connection", lambda: FakeConnectionContext())

    stats = daily_report.fetch_daily_user_chat_stats(
        target_date=date(2026, 7, 7),
        table_name="chat_messages",
        timezone=ZoneInfo("Asia/Shanghai"),
    )

    start, end = captured["params"]
    assert start == datetime(2026, 7, 7, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert end == datetime(2026, 7, 8, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert stats == [
        daily_report.DailyUserChatStat("Alice", 2),
        daily_report.DailyUserChatStat("未识别用户", 1),
    ]


def test_send_daily_report_uses_union_id(monkeypatch) -> None:
    sent: dict[str, object] = {}

    def fake_fetch(*, target_date=None):
        assert target_date == date(2026, 7, 7)
        return [daily_report.DailyUserChatStat("Alice", 2)]

    async def fake_send(**kwargs):
        sent.update(kwargs)
        return "om_test_message"

    monkeypatch.setattr(daily_report, "fetch_daily_user_chat_stats", fake_fetch)
    monkeypatch.setattr(daily_report.feishu, "_send_feishu_markdown_message_to_receive_id", fake_send)
    monkeypatch.setattr(
        daily_report,
        "settings",
        SimpleNamespace(
            daily_report_enabled=True,
            daily_report_target_union_id="on_default",
            daily_report_target_session_id="oc_default",
            daily_report_timezone="Asia/Shanghai",
            postgres_chat_table="chat_messages",
        ),
    )

    result = asyncio.run(
        daily_report.send_daily_report(
            target_union_id="on_ebc25d5669cabb3440819db2cfaa5c7c",
            target_session_id="oc_b4325718ab22291bc7625ebd63d6f915",
            target_date=date(2026, 7, 7),
        )
    )

    assert sent["receive_id"] == "on_ebc25d5669cabb3440819db2cfaa5c7c"
    assert sent["receive_id_type"] == "union_id"
    assert "Alice：2 次" in sent["markdown_text"]
    assert result["ok"] is True
    assert result["target_union_id"] == "on_ebc25d5669cabb3440819db2cfaa5c7c"


def test_send_daily_report_skips_when_disabled(monkeypatch) -> None:
    async def fail_send(**_kwargs):
        raise AssertionError("disabled report should not send Feishu message")

    monkeypatch.setattr(daily_report.feishu, "_send_feishu_markdown_message_to_receive_id", fail_send)
    monkeypatch.setattr(
        daily_report,
        "settings",
        SimpleNamespace(
            daily_report_enabled=False,
            daily_report_target_union_id="on_default",
            daily_report_target_session_id="oc_default",
            daily_report_timezone="Asia/Shanghai",
            postgres_chat_table="chat_messages",
        ),
    )

    result = asyncio.run(daily_report.send_daily_report())

    assert result == {
        "ok": False,
        "skipped": True,
        "reason": "daily_report_disabled",
        "message_id": None,
    }


def test_seconds_until_next_daily_run_rolls_to_tomorrow_after_nine() -> None:
    seconds = daily_report.seconds_until_next_daily_run(
        now=datetime(2026, 7, 7, 9, 1, tzinfo=timezone.utc),
        hour=9,
        minute=0,
        timezone=timezone.utc,
    )

    assert seconds == 23 * 60 * 60 + 59 * 60
