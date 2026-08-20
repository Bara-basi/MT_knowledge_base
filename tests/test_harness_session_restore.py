from __future__ import annotations

import json
from pathlib import Path

from app.services.harness import _load_persisted_history, _prompt_with_restored_history


def _write_event(stream, event_type: str, data: dict) -> None:
    stream.write(json.dumps({"type": event_type, "data": data}, ensure_ascii=False) + "\n")


def test_restart_history_uses_the_same_jsonl_and_does_not_nest_restored_context(tmp_path: Path) -> None:
    session_id = "mtsco-stable-session"
    session_file = tmp_path / "cwd" / session_id / "session.jsonl"
    session_file.parent.mkdir(parents=True)
    with session_file.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "session", "id": session_id}) + "\n")
        _write_event(stream, "user/message", {"content": [{"type": "text", "text": "第一问"}]})
        _write_event(stream, "assistant/message", {"message": {"content": [{"type": "text", "text": "第一答"}]}})
        _write_event(
            stream,
            "user/message",
            {"content": [{"type": "text", "text": (
                "忽略这一段包装\n"
                "<mtsco-restored-history>\n用户：第一问\n</mtsco-restored-history>\n"
                "<mtsco-current-message>\n第二问\n</mtsco-current-message>"
            )}]},
        )

    history = _load_persisted_history(tmp_path, session_id)

    assert history == [("用户", "第一问"), ("助手", "第一答"), ("用户", "第二问")]
    prompt = _prompt_with_restored_history(tmp_path, session_id, "第三问")
    assert "用户：第一问" in prompt
    assert "用户：第二问" in prompt
    assert "第三问" in prompt
    assert prompt.count("<mtsco-restored-history>") == 1


def test_restart_history_ignores_other_session_files(tmp_path: Path) -> None:
    for session_id, text in (("mtsco-current", "当前会话"), ("mtsco-other", "其他会话")):
        session_file = tmp_path / session_id / "session.jsonl"
        session_file.parent.mkdir(parents=True)
        with session_file.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({"type": "session", "id": session_id}) + "\n")
            _write_event(stream, "user/message", {"content": [{"type": "text", "text": text}]})

    assert _load_persisted_history(tmp_path, "mtsco-current") == [("用户", "当前会话")]


def test_restart_prefers_completed_application_transcript(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "app.services.harness._load_chat_record_history",
        lambda _session_id: [("用户", "代码 7301"), ("助手", "收到 7301")],
    )

    prompt = _prompt_with_restored_history(
        tmp_path,
        "mtsco-not-written-by-runtime",
        "代码 5312",
        internal_session_id="stable-session",
    )

    assert "用户：代码 7301" in prompt
    assert "助手：收到 7301" in prompt
    assert "代码 5312" in prompt
