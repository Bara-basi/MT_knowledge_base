from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from app.harness import mcp_server
from app.workers import harness_scheduler


def test_archived_session_log_is_deleted_only_for_matching_session(monkeypatch, tmp_path: Path) -> None:
    session_id = "archived-session"
    session_dir = tmp_path / "runtime" / f"mtsco-{session_id}"
    session_dir.mkdir(parents=True)
    (session_dir / "session.jsonl").write_text(
        json.dumps({"type": "session", "id": f"mtsco-{session_id}"}) + "\n",
        encoding="utf-8",
    )
    other_dir = tmp_path / "runtime" / "mtsco-active-session"
    other_dir.mkdir()
    (other_dir / "session.jsonl").write_text(
        json.dumps({"type": "session", "id": "mtsco-active-session"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(harness_scheduler, "settings", SimpleNamespace(harness_session_root=str(tmp_path)))

    assert harness_scheduler._delete_archived_session_log(session_id) == 1
    assert not session_dir.exists()
    assert other_dir.exists()


def test_conversation_summary_reads_only_current_users_latest_memory(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_list_harness_memories(**kwargs):
        calls.update(kwargs)
        return [{
            "internal_session_id": "session-id",
            "topic": "产品规格",
            "summary": "# 产品规格\n\n已确认规格。",
            "started_at": "2026-08-20T01:00:00Z",
            "ended_at": "2026-08-20T01:05:00Z",
        }]

    monkeypatch.setattr("app.db.postgres.list_harness_memories", fake_list_harness_memories)
    monkeypatch.setattr(mcp_server, "USER_ID", "current-open-id")

    result = mcp_server.call("conversation_summary", {"scope": "latest"})

    assert result["isError"] is False
    assert calls["user_id"] == "current-open-id"
    assert calls["limit"] == 1
    assert json.loads(result["content"][0]["text"])[0]["summary"] == "# 产品规格\n\n已确认规格。"


def test_excerpt_search_returns_bounded_matching_turns(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server,
        "_read_archived_turns",
        lambda _memory: [{"question": "上次的不锈钢规格是什么？", "answer": "316L，厚度 2mm。"}],
    )

    records = mcp_server._excerpt_records(
        [{"internal_session_id": "session-id", "topic": "规格", "ended_at": "2026-08-20"}],
        "不锈钢规格",
        3,
    )

    assert records == [{
        "session_id": "session-id",
        "topic": "规格",
        "ended_at": "2026-08-20",
        "question_excerpt": "上次的不锈钢规格是什么？",
        "answer_excerpt": "316L，厚度 2mm。",
    }]
