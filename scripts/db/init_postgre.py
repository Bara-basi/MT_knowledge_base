from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines before importing app settings."""

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_file(PROJECT_ROOT / ".env")

from app.db.postgres import (  # noqa: E402
    check_postgres_health,
    ensure_chat_messages_table,
    ensure_conversation_topics_table,
    ensure_external_chat_messages_table,
    insert_chat_message,
)


SAMPLE_MESSAGE = {
    "question": "请说明如何批量导入培训题目并发布考试。",
    "user_id": "on_ebc25d5669cabb3440819db2cfaa5c7c",
    "session_id": "oc_161f3d51b1e5caf056812ab5312f6cb6",
    "conversation_id": "oc_161f3d51b1e5caf056812ab5312f6cb6",
    "answer": "初始化测试回答",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize PostgreSQL for MTSCO KB.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check PostgreSQL connectivity; do not create tables.",
    )
    parser.add_argument(
        "--insert-sample",
        action="store_true",
        help="Insert one sample chat row after ensuring the table exists.",
    )
    parser.add_argument(
        "--table-name",
        default=None,
        help="Override the chat table name. Defaults to POSTGRES_CHAT_TABLE.",
    )
    args = parser.parse_args()

    if args.check_only:
        result = check_postgres_health()
    else:
        result = {
            "chat_messages": ensure_chat_messages_table(table_name=args.table_name),
            "chat_messages_external": ensure_external_chat_messages_table(),
            "conversation_topics": ensure_conversation_topics_table(),
        }
        if args.insert_sample:
            result["sample_row"] = insert_chat_message(
                table_name=args.table_name,
                **SAMPLE_MESSAGE,
            )

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
