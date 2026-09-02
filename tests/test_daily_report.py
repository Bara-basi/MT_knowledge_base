from __future__ import annotations

import asyncio
from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from app.services import daily_report
from app.services.usage_report import UsageReportData, UsageUserStat


def test_default_report_date_uses_previous_day_in_report_timezone() -> None:
    report_date = daily_report.default_report_date(
        datetime(2026, 7, 8, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    )

    assert report_date == date(2026, 7, 7)


def test_build_daily_report_text_lists_previous_day_user_counts() -> None:
    text = daily_report.build_daily_report_text(
        [
            daily_report.DailyUserChatStat("Alice", 3),
            daily_report.DailyUserChatStat("Bob", 1),
        ],
        target_date=date(2026, 7, 7),
        generated_at=datetime(2026, 7, 8, 9, 5, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert "📊 **MTSCO 知识库 · 日使用简报**" in text
    assert "2026-07-07（昨日）" in text
    assert "💬 **4** 条提问" in text
    assert "🥇 Alice" in text
    assert "🥈 Bob" in text
    assert "Top 3 用户贡献占比：**100.0%**" in text
    assert "统计范围：全部用户" in text
    assert "PostgreSQL" not in text
    assert "chat_messages" not in text


def test_fetch_daily_user_chat_stats_queries_target_day(monkeypatch) -> None:
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


def test_fetch_daily_user_chat_stats_accepts_braced_department(monkeypatch) -> None:
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
            return []

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

    class FakeContext:
        def __enter__(self):
            return FakeConnection()

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(daily_report, "ensure_chat_messages_table", lambda _name=None: None)
    monkeypatch.setattr(daily_report, "postgres_connection", lambda: FakeContext())

    daily_report.fetch_daily_user_chat_stats(
        target_date=date(2026, 7, 7),
        department_names="{迈拓思学园}",
    )

    assert captured["params"][-1] == ["迈拓思学园"]
    assert "department_names" in str(captured["query"])


def test_send_daily_report_uses_default_previous_day(monkeypatch) -> None:
    sent: dict[str, object] = {}
    fetched: dict[str, object] = {}

    def fake_fetch(*, target_date=None):
        fetched["target_date"] = target_date
        return UsageReportData(
            user_stats=[UsageUserStat("Alice", 2, 2)],
            total_questions=2,
            active_users=1,
            answered_questions=2,
        )

    async def fake_send(**kwargs):
        sent.update(kwargs)
        return "om_test_message"

    monkeypatch.setattr(daily_report, "default_report_date", lambda *_args, **_kwargs: date(2026, 7, 7))
    monkeypatch.setattr(daily_report, "fetch_daily_report_data", fake_fetch)
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
        )
    )

    assert fetched["target_date"] == date(2026, 7, 7)
    assert sent["receive_id"] == "on_ebc25d5669cabb3440819db2cfaa5c7c"
    assert sent["receive_id_type"] == "union_id"
    assert "日使用简报" in sent["markdown_text"]
    assert "🥇 Alice" in sent["markdown_text"]
    assert result["ok"] is True
    assert result["report_date"] == date(2026, 7, 7)
    assert result["report_data"].total_questions == 2


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


def test_send_daily_report_sends_to_open_id(monkeypatch) -> None:
    sent: dict[str, object] = {}

    async def fake_send(**kwargs):
        sent.update(kwargs)
        return "om_test_message"

    monkeypatch.setattr(daily_report, "fetch_daily_report_data", lambda **_kwargs: UsageReportData())
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

    result = asyncio.run(daily_report.send_daily_report(target_open_id="ou_requester"))

    assert sent["receive_id"] == "ou_requester"
    assert sent["receive_id_type"] == "open_id"
    assert result["target_union_id"] == ""
    assert result["target_open_id"] == "ou_requester"
