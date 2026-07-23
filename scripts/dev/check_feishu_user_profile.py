from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.v1.feishu import fetch_feishu_user_profile_by_union_id  # noqa: E402


def _mask_identifier(value: str | None) -> str | None:
    if not value or len(value) <= 8:
        return value
    return f"{value[:4]}...{value[-4:]}"


def _serialize_profile(profile: Any, *, show_ids: bool) -> dict[str, Any]:
    data = asdict(profile)
    data["department_ids"] = list(data["department_ids"])
    data["department_names"] = list(data["department_names"])
    data["returned_fields"] = list(data["returned_fields"])
    if not show_ids:
        for key in ("union_id", "user_id", "open_id"):
            data[key] = _mask_identifier(data[key])
        data["department_ids"] = [
            _mask_identifier(item) for item in data["department_ids"]
        ]
    data["organization_fields_ready"] = bool(
        data["department_field_available"]
        and data["department_names"]
        and data["job_title_field_available"]
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve a Feishu union_id into user, department, and job fields."
    )
    parser.add_argument("--union-id", required=True, help="Feishu union_id to inspect.")
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Print complete Feishu identifiers instead of masking them.",
    )
    args = parser.parse_args()

    profile = asyncio.run(fetch_feishu_user_profile_by_union_id(args.union_id))
    print(
        json.dumps(
            _serialize_profile(profile, show_ids=args.show_ids),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
