from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import HTTPException


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines before importing application settings."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_file(PROJECT_ROOT / ".env")

from app.api.v1.feishu import (  # noqa: E402
    FeishuUserProfile,
    fetch_feishu_user_profile_by_union_id,
)
from app.db.postgres import (  # noqa: E402
    ensure_chat_user_profile_columns,
    list_chat_user_profile_backfill_candidates,
    update_chat_user_profile,
)


def mask_identifier(value: str) -> str:
    value = str(value).strip()
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-4:]}"


def error_message(exc: Exception) -> str:
    detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
    if isinstance(detail, (dict, list)):
        return json.dumps(detail, ensure_ascii=False, default=str)[:500]
    return str(detail)[:500]


async def fetch_profiles(
    candidates: list[dict[str, Any]],
    *,
    concurrency: int,
) -> tuple[list[tuple[dict[str, Any], FeishuUserProfile]], list[dict[str, Any]]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def fetch_one(candidate: dict[str, Any]) -> tuple[dict[str, Any], FeishuUserProfile | None, str | None]:
        union_id = str(candidate["union_id"])
        try:
            async with semaphore:
                profile = await fetch_feishu_user_profile_by_union_id(union_id)
            return candidate, profile, None
        except Exception as exc:  # One inaccessible user must not stop the batch.
            return candidate, None, error_message(exc)

    results = await asyncio.gather(*(fetch_one(candidate) for candidate in candidates))
    succeeded = [(candidate, profile) for candidate, profile, _ in results if profile is not None]
    failures = [
        {
            "union_id": mask_identifier(str(candidate["union_id"])),
            "row_count": int(candidate["row_count"]),
            "error": error,
        }
        for candidate, profile, error in results
        if profile is None
    ]
    return succeeded, failures


async def run_backfill(args: argparse.Namespace) -> dict[str, Any]:
    if not args.dry_run:
        await asyncio.to_thread(ensure_chat_user_profile_columns, table_name=args.table_name)
    elif not args.force:
        # A dry run is read-only and must also work before the new columns exist.
        args.force = True

    candidates = await asyncio.to_thread(
        list_chat_user_profile_backfill_candidates,
        table_name=args.table_name,
        union_ids=args.union_id,
        force=args.force,
    )
    succeeded, failures = await fetch_profiles(candidates, concurrency=args.concurrency)

    updated_rows = 0
    departments: Counter[str] = Counter()
    preview: list[dict[str, Any]] = []
    for candidate, profile in succeeded:
        departments.update(profile.department_names or ("未解析部门",))
        preview.append(
            {
                "union_id": mask_identifier(profile.union_id),
                "name": profile.name,
                "departments": list(profile.department_names),
                "job_title": profile.job_title,
                "employee_type": profile.employee_type,
                "row_count": int(candidate["row_count"]),
            }
        )
        if not args.dry_run:
            updated_rows += await asyncio.to_thread(
                update_chat_user_profile,
                union_id=str(candidate["union_id"]),
                feishu_user_id=profile.user_id,
                feishu_open_id=profile.open_id,
                user_name=profile.name,
                department_ids=profile.department_ids,
                department_names=profile.department_names,
                job_title=profile.job_title,
                employee_type=profile.employee_type,
                table_name=args.table_name,
            )

    return {
        "mode": "dry-run" if args.dry_run else "write",
        "table_name": args.table_name or "POSTGRES_CHAT_TABLE",
        "candidate_users": len(candidates),
        "succeeded_users": len(succeeded),
        "failed_users": len(failures),
        "updated_rows": updated_rows,
        "department_user_counts": dict(sorted(departments.items())),
        "profiles": preview,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill Feishu user, department, and job data into historical chat messages."
    )
    parser.add_argument("--table-name", default=None, help="Override POSTGRES_CHAT_TABLE.")
    parser.add_argument(
        "--union-id",
        action="append",
        default=None,
        help="Only process this union_id; may be supplied multiple times.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch and preview without DB writes.")
    parser.add_argument("--force", action="store_true", help="Refresh users already backfilled.")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent Feishu requests (1-20).")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 20:
        parser.error("--concurrency must be between 1 and 20")
    result = asyncio.run(run_backfill(args))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
