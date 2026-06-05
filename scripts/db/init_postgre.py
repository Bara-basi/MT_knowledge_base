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
    insert_chat_message,
)


SAMPLE_MESSAGE = {
    "question": (
        "我下周要组织一批新员工参加入职培训，培训结束后需要进行线上考核。作为管理员，"
        "我需要完成以下几个关联任务：首先，我需要创建一个新的题库专门用来存放这次培训的考核题目，"
        "并且希望能够快速批量导入而不是一道道手动录入；其次，我要基于这个题库发布一场正式的考试，"
        "有几个硬性要求：为了防止员工重复刷分，必须限制每人只能考一次，如果试卷里的题目数量不足系统"
        "要自动禁止考试不能凑数，而且考试中必须包含简答题需要人工批改而不是系统自动判分，还要指定"
        "具体的阅卷负责人；最后，我需要把这场考试添加到现有的“新员工入职培训”学习项目中作为结业考核环节。"
        "请详细说明完成这些任务的每个步骤的完整操作路径、关键设置点的具体选项和按钮位置，以及根据操作指引，"
        "哪些地方最容易被忽略或操作失误导致整个设置无效？"
    ),
    "user_id": "on_ebc25d5669cabb3440819db2cfaa5c7c",
    "session_id": "oc_161f3d51b1e5caf056812ab5312f6cb6",
    "conversation_id": "oc_161f3d51b1e5caf056812ab5312f6cb6",
    "answer": "初始化测试回答",
    "fallback": False,
    "reason": "",
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
        result = ensure_chat_messages_table(table_name=args.table_name)
        if args.insert_sample:
            result["sample_row"] = insert_chat_message(
                table_name=args.table_name,
                **SAMPLE_MESSAGE,
            )

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
