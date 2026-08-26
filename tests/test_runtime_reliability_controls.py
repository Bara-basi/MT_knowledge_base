from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

from app.services import harness as harness_service
from app.services.harness import _advisory_key, _compose_task_prompt, _latest_context_tokens
from app.services.metrics import observe_http, render_prometheus
from app.services.parser.powerpoint_parser import _build_items
from app.services.parser.word_parser import _clean_document_items
from scripts.evaluation.run_basic_eval import score_answer
from app.workers.harness_scheduler import _session_lock_key


def test_latest_context_tokens_uses_newest_assistant_usage_and_cache_tokens() -> None:
    result = SimpleNamespace(
        events=[
            {"type": "assistant/message", "data": {"usage": {"inputTokens": 10}}},
            {"type": "tool/result", "data": {"usage": {"inputTokens": 999}}},
            {
                "type": "assistant/message",
                "data": {
                    "usage": {
                        "inputTokens": 60_000,
                        "cacheReadTokens": 29_000,
                        "cacheWriteTokens": 1_000,
                        "outputTokens": 500,
                    }
                },
            },
        ]
    )

    assert _latest_context_tokens(result) == 90_000


def test_scheduler_and_harness_share_the_same_session_lock() -> None:
    session_id = "00000000-0000-0000-0000-000000000001"

    assert _session_lock_key(session_id) == _advisory_key(f"harness-session:{session_id}")


def test_graceful_shutdown_closes_cached_harness_processes(monkeypatch) -> None:
    closed: list[str] = []

    class FakeHarness:
        def close(self) -> None:
            closed.append("closed")

    monkeypatch.setattr(
        harness_service,
        "_LOCAL_HARNESSES",
        {"session": (FakeHarness(), threading.RLock())},
    )

    assert harness_service.close_local_harnesses() == 1
    assert closed == ["closed"]
    assert harness_service._LOCAL_HARNESSES == {}


def test_harness_turn_failure_extracts_transport_error() -> None:
    result = {
        "finish_reason": "error",
        "events": [
            {
                "type": "turn/end",
                "data": {
                    "reason": {
                        "kind": "error",
                        "error": {"code": "TRANSPORT", "message": "Connection error."},
                    }
                },
            }
        ],
    }

    assert harness_service._harness_turn_failure(result) == "TRANSPORT: Connection error."


def test_harness_turn_failure_accepts_successful_endings() -> None:
    assert harness_service._harness_turn_failure({"finish_reason": "completed"}) == ""
    assert harness_service._harness_turn_failure({"finish_reason": "max-tokens"}) == ""


def test_harness_session_sequence_validation_handles_compressed_events(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    session_file.write_text(
        "\n".join(
            [
                '{"type":"session","id":"mtsco-test"}',
                '{"type":"event","seq":0}',
                '{"type":"text-chunks","seq0":1,"data":{"texts":["a","b"]}}',
                '{"type":"event","seq":3}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert harness_service._session_log_sequence_is_valid(session_file) is True


def test_corrupt_harness_session_is_moved_outside_active_root(tmp_path: Path) -> None:
    session_root = tmp_path / "harness_sessions"
    session_dir = session_root / "workspace" / "mtsco-test"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "session.jsonl"
    session_file.write_text(
        "\n".join(
            [
                '{"type":"session","id":"mtsco-test"}',
                '{"type":"event","seq":0}',
                '{"type":"event","seq":2}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    destination = harness_service._quarantine_corrupt_session_log(
        session_root,
        "mtsco-test",
    )

    assert destination is not None
    assert not session_dir.exists()
    assert (destination / "session.jsonl").exists()
    assert session_root not in destination.parents


def test_external_task_fields_are_preserved_in_harness_prompt() -> None:
    prompt = _compose_task_prompt(
        question="请评分",
        additional_system_prompt="只返回合法 JSON",
        task_input='{"price":32}',
        metadata={"service": "crm"},
    )

    assert "请评分" in prompt
    assert "只返回合法 JSON" in prompt
    assert '{"price":32}' in prompt
    assert '"service":"crm"' in prompt


def test_harness_composition_enables_rooted_read_only_filesystem() -> None:
    cordis = (Path(__file__).parents[1] / "app" / "harness" / "cordis.yml").read_text(
        encoding="utf-8"
    )

    assert "rooted: true" in cordis
    assert "readOnly: true" in cordis
    assert "name: '@deepseek-ai/dsh-tool-bash'" not in cordis


def test_harness_composition_enables_bounded_provider_retries() -> None:
    cordis = (Path(__file__).parents[1] / "app" / "harness" / "cordis.yml").read_text(
        encoding="utf-8"
    )

    assert cordis.count("name: '@deepseek-ai/dsh-llm-retry'") == 1

    project_root = Path(__file__).parents[1]
    windows_installer = (project_root / "scripts" / "harness" / "install-windows.ps1").read_text(
        encoding="utf-8"
    )
    linux_installer = (project_root / "scripts" / "harness" / "install-linux.sh").read_text(
        encoding="utf-8"
    )
    assert '"dsh-llm-retry" = "packages\\llm\\llm-retry"' in windows_installer
    assert "dsh-llm-retry:packages/llm/llm-retry" in linux_installer


def test_word_cleaning_keeps_reference_sections_and_source_links() -> None:
    items = [
        {"type": "paragraph", "style": "标题 1", "text": "参考文献"},
        {"type": "paragraph", "style": "正文", "text": "资料来源：https://example.test/spec"},
    ]

    assert _clean_document_items(items) == items


def test_powerpoint_build_keeps_the_final_slide() -> None:
    slides = [
        {
            "slide_index": 1,
            "elements": [
                {"type": "paragraph", "style": "正文", "text": "第一页正文", "font_size": 12}
            ],
        },
        {
            "slide_index": 2,
            "elements": [
                {"type": "paragraph", "style": "正文", "text": "最后一页关键数据", "font_size": 12}
            ],
        },
    ]

    items = _build_items(slides)

    assert any(item.get("text") == "最后一页关键数据" for item in items)


def test_metrics_render_process_and_database_signals() -> None:
    observe_http(method="GET", path="/health/ready", status=200, duration_seconds=0.25)

    rendered = render_prometheus(
        answer_jobs={"counts": {"queued": 2}, "oldest_queued_seconds": 3.5},
        harness_sessions={"active": 1},
    )

    assert 'mtsco_http_requests_total{method="GET"' in rendered
    assert 'mtsco_answer_jobs{status="queued"} 2' in rendered
    assert "mtsco_answer_job_oldest_queued_seconds 3.500" in rendered
    assert 'mtsco_harness_sessions{status="active"} 1' in rendered


def test_basic_eval_checks_recall_and_internal_information_leaks() -> None:
    case = {
        "id": "smoke",
        "category": "basic",
        "expected_keywords": ["316L", "2mm"],
        "minimum_keyword_recall": 1.0,
    }

    passed = score_answer(case, "建议使用 316L，厚度 2mm。", 100)
    leaked = score_answer(case, "316L 2mm，内部路径为 data/processing。", 100)

    assert passed.passed is True
    assert leaked.passed is False
    assert "data/processing" in leaked.violations
