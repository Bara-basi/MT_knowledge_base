from __future__ import annotations

import argparse
import asyncio

from app.api.v1.feishu import FeishuUserProfile
from scripts.db import backfill_chat_user_profiles as backfill


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "table_name": "chat_messages",
        "union_id": None,
        "dry_run": False,
        "force": False,
        "concurrency": 2,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _profile(union_id: str) -> FeishuUserProfile:
    return FeishuUserProfile(
        union_id=union_id,
        user_id="tenant-user",
        open_id="open-user",
        name="测试用户",
        department_ids=("dept-1",),
        department_names=("迈拓思学园",),
        job_title="学员",
        employee_type=1,
        department_field_available=True,
        job_title_field_available=True,
        returned_fields=("department_ids", "job_title"),
    )


def test_fetch_profiles_keeps_processing_after_one_failure(monkeypatch) -> None:
    async def fake_fetch(union_id: str) -> FeishuUserProfile:
        if union_id == "union-bad":
            raise RuntimeError("no user authority")
        return _profile(union_id)

    monkeypatch.setattr(backfill, "fetch_feishu_user_profile_by_union_id", fake_fetch)
    candidates = [
        {"union_id": "union-good-1234", "row_count": 3},
        {"union_id": "union-bad", "row_count": 2},
    ]

    succeeded, failures = asyncio.run(backfill.fetch_profiles(candidates, concurrency=2))

    assert len(succeeded) == 1
    assert succeeded[0][1].name == "测试用户"
    assert failures == [
        {"union_id": "unio…-bad", "row_count": 2, "error": "no user authority"}
    ]


def test_run_backfill_updates_every_historical_row(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append((func.__name__, kwargs))
        if func is backfill.list_chat_user_profile_backfill_candidates:
            return [{"union_id": "union-user-1234", "row_count": 4}]
        if func is backfill.update_chat_user_profile:
            return 4
        return {"ok": True}

    async def fake_fetch_profiles(candidates, *, concurrency):
        return [(candidates[0], _profile("union-user-1234"))], []

    monkeypatch.setattr(backfill.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(backfill, "fetch_profiles", fake_fetch_profiles)

    result = asyncio.run(backfill.run_backfill(_args()))

    assert result["updated_rows"] == 4
    assert result["department_user_counts"] == {"迈拓思学园": 1}
    update_call = next(kwargs for name, kwargs in calls if name == "update_chat_user_profile")
    assert update_call["department_names"] == ("迈拓思学园",)
    assert update_call["employee_type"] == 1


def test_dry_run_does_not_change_database(monkeypatch) -> None:
    called: list[str] = []

    async def fake_to_thread(func, *args, **kwargs):
        called.append(func.__name__)
        if func is backfill.list_chat_user_profile_backfill_candidates:
            assert kwargs["force"] is True
            return []
        raise AssertionError(f"Unexpected write: {func.__name__}")

    monkeypatch.setattr(backfill.asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(backfill.run_backfill(_args(dry_run=True)))

    assert result["mode"] == "dry-run"
    assert called == ["list_chat_user_profile_backfill_candidates"]


def test_mask_identifier_does_not_print_full_union_id() -> None:
    assert backfill.mask_identifier("on_1234567890abcdef") == "on_1…cdef"
