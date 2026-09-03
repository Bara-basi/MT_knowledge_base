from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from app.services import weekly_report
from app.services.usage_report import UsageReportData, UsageUserStat


SHANGHAI = ZoneInfo("Asia/Shanghai")


def test_get_report_timezone_falls_back_for_an_invalid_zoneinfo_key(monkeypatch) -> None:
    monkeypatch.setattr(
        weekly_report,
        "settings",
        SimpleNamespace(weekly_report_timezone="Asia/"),
    )

    assert weekly_report.get_report_timezone().key == "Asia/Shanghai"


def test_default_report_end_uses_last_friday_when_run_on_monday() -> None:
    report_end = weekly_report.default_report_end(
        datetime(2026, 7, 20, 9, 0, tzinfo=SHANGHAI)
    )

    assert report_end == datetime(2026, 7, 17, 17, 30, tzinfo=SHANGHAI)


def test_default_report_end_uses_previous_friday_before_friday_cutoff() -> None:
    report_end = weekly_report.default_report_end(
        datetime(2026, 7, 17, 10, 0, tzinfo=SHANGHAI)
    )

    assert report_end == datetime(2026, 7, 10, 17, 30, tzinfo=SHANGHAI)


def test_build_weekly_report_text_lists_user_counts() -> None:
    text = weekly_report.build_weekly_report_text(
        [
            weekly_report.WeeklyUserChatStat("Alice", 5),
            weekly_report.WeeklyUserChatStat("Bob", 2),
        ],
        report_end=datetime(2026, 7, 17, 17, 30, tzinfo=SHANGHAI),
    )

    assert "📊 **MTSCO 知识库 · 周使用简报**" in text
    assert "2026-07-10 17:30 — 2026-07-17 17:30" in text
    assert "💬 **7** 条提问" in text
    assert "🥇 Alice" in text
    assert "🥈 Bob" in text
    assert "Top 3 用户贡献占比：**100.0%**" in text
    assert "统计范围：全部用户" in text
    assert "PostgreSQL" not in text
    assert "chat_messages" not in text


def test_fetch_weekly_user_chat_stats_queries_half_open_week(monkeypatch) -> None:
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
            return [{"user_name": "Alice", "message_count": 4}]

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

    class FakeConnectionContext:
        def __enter__(self):
            return FakeConnection()

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(weekly_report, "ensure_chat_messages_table", lambda table_name=None: None)
    monkeypatch.setattr(weekly_report, "postgres_connection", lambda: FakeConnectionContext())

    stats = weekly_report.fetch_weekly_user_chat_stats(
        report_end=datetime(2026, 7, 17, 17, 30, tzinfo=SHANGHAI),
        table_name="chat_messages",
        timezone=SHANGHAI,
    )

    assert captured["params"] == (
        datetime(2026, 7, 10, 17, 30, tzinfo=SHANGHAI),
        datetime(2026, 7, 17, 17, 30, tzinfo=SHANGHAI),
    )
    assert stats == [weekly_report.WeeklyUserChatStat("Alice", 4)]


def test_send_weekly_report_uses_default_completed_week(monkeypatch) -> None:
    sent: dict[str, object] = {}
    fetched: dict[str, object] = {}
    report_end = datetime(2026, 7, 17, 17, 30, tzinfo=SHANGHAI)

    def fake_fetch(*, report_end=None):
        fetched["report_end"] = report_end
        return UsageReportData(
            user_stats=[UsageUserStat("Alice", 3, 3)],
            total_questions=3,
            active_users=1,
            answered_questions=3,
        )

    async def fake_send(**kwargs):
        sent.update(kwargs)
        return "om_test_message"

    monkeypatch.setattr(weekly_report, "default_report_end", lambda *_args, **_kwargs: report_end)
    monkeypatch.setattr(weekly_report, "fetch_weekly_report_data", fake_fetch)
    monkeypatch.setattr(
        weekly_report.feishu,
        "_send_feishu_markdown_message_to_receive_id",
        fake_send,
    )
    monkeypatch.setattr(
        weekly_report,
        "settings",
        SimpleNamespace(
            weekly_report_enabled=True,
            weekly_report_target_union_id="on_default",
            weekly_report_target_session_id="oc_default",
            weekly_report_timezone="Asia/Shanghai",
            postgres_chat_table="chat_messages",
        ),
    )

    result = asyncio.run(weekly_report.send_weekly_report(target_union_id="on_target"))

    assert fetched["report_end"] == report_end
    assert sent["receive_id"] == "on_target"
    assert sent["receive_id_type"] == "union_id"
    assert "2026-07-10 17:30 — 2026-07-17 17:30" in sent["markdown_text"]
    assert result["ok"] is True
    assert result["report_start"] == datetime(2026, 7, 10, 17, 30, tzinfo=SHANGHAI)
    assert result["report_end"] == report_end
    assert result["report_data"].total_questions == 3


def test_send_weekly_report_force_bypasses_disabled_setting(monkeypatch) -> None:
    async def fake_send(**_kwargs):
        return "om_test_message"

    monkeypatch.setattr(
        weekly_report,
        "fetch_weekly_report_data",
        lambda **_kwargs: UsageReportData(),
    )
    monkeypatch.setattr(
        weekly_report.feishu,
        "_send_feishu_markdown_message_to_receive_id",
        fake_send,
    )
    monkeypatch.setattr(
        weekly_report,
        "settings",
        SimpleNamespace(
            weekly_report_enabled=False,
            weekly_report_target_union_id="on_default",
            weekly_report_target_session_id="",
            weekly_report_timezone="Asia/Shanghai",
            postgres_chat_table="chat_messages",
        ),
    )

    result = asyncio.run(weekly_report.send_weekly_report(force=True))

    assert result["ok"] is True


def test_send_weekly_report_sends_to_open_id(monkeypatch) -> None:
    sent: dict[str, object] = {}

    async def fake_send(**kwargs):
        sent.update(kwargs)
        return "om_test_message"

    monkeypatch.setattr(weekly_report, "fetch_weekly_report_data", lambda **_kwargs: UsageReportData())
    monkeypatch.setattr(weekly_report.feishu, "_send_feishu_markdown_message_to_receive_id", fake_send)
    monkeypatch.setattr(
        weekly_report,
        "settings",
        SimpleNamespace(
            weekly_report_enabled=True,
            weekly_report_target_union_id="on_default",
            weekly_report_target_session_id="oc_default",
            weekly_report_timezone="Asia/Shanghai",
            postgres_chat_table="chat_messages",
        ),
    )

    result = asyncio.run(weekly_report.send_weekly_report(target_open_id="ou_requester"))

    assert sent["receive_id"] == "ou_requester"
    assert sent["receive_id_type"] == "open_id"
    assert result["target_union_id"] == ""
    assert result["target_open_id"] == "ou_requester"
