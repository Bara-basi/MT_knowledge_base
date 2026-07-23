from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.weekly_report import run_weekly_report_loop, send_weekly_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Send MTSCO knowledge base weekly usage report.")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep running and send every Friday at 17:35.",
    )
    parser.add_argument("--union-id", default=None, help="Target Feishu union_id.")
    parser.add_argument("--open-id", default=None, help="Deprecated alias for --union-id.")
    parser.add_argument("--session-id", default=None, help="Target Feishu chat/session id.")
    parser.add_argument(
        "--department",
        action="append",
        default=None,
        help="Only include this department; repeat for multiple departments. Braces are optional.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Send once even when WEEKLY_REPORT_ENABLED=false.",
    )
    args = parser.parse_args()

    if args.loop:
        asyncio.run(run_weekly_report_loop(department_names=args.department))
        return

    result = asyncio.run(
        send_weekly_report(
            target_union_id=args.union_id,
            target_open_id=args.open_id,
            target_session_id=args.session_id,
            department_names=args.department,
            force=args.force,
        )
    )
    if result.get("skipped"):
        print(f"skipped=true reason={result['reason']}")
        return
    print(result["text"])
    print(f"sent={result['ok']} message_id={result['message_id']}")


if __name__ == "__main__":
    main()
