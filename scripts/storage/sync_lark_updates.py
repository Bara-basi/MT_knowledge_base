from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.storage.lark_script_credentials import use_local_lark_credentials  # noqa: E402


use_local_lark_credentials()

from app.services.lark_document_sync import (  # noqa: E402
    DEFAULT_VECTOR_SRC,
    PACIFIC_FIXED_TZ,
    next_utc_minus_8_midnight,
    run_daily_loop,
    scan_lark_updates,
)


def main() -> int:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "Scan Feishu/Lark documents from vector_src.json and re-ingest changed "
            "documents. Default mode is one scan; use --loop for daily UTC-8 00:00."
        )
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_VECTOR_SRC),
        help="Path to vector_src.json.",
    )
    parser.add_argument(
        "--document-name",
        help="Only scan/update one document name from ingestion_registry.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Process matching documents even when Lark updated_at is not newer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect candidates and compare hashes without writing MinIO/Milvus.",
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=3,
        help="Max concurrent image description calls during re-ingestion. Default: 3.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run forever, triggering at UTC-8 00:00 every day.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one scan and exit. This is the default unless --loop is set.",
    )
    parser.add_argument(
        "--print-next-run",
        action="store_true",
        help="Print the next UTC-8 midnight run time and exit.",
    )
    args = parser.parse_args()

    if args.print_next_run:
        next_run = next_utc_minus_8_midnight()
        print(
            json.dumps(
                {
                    "now_utc": datetime.now(timezone.utc),
                    "next_run_utc": next_run,
                    "next_run_utc_minus_8": next_run.astimezone(PACIFIC_FIXED_TZ),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0

    if args.loop and not args.document_name and not args.force and not args.dry_run:
        print("Starting daily Lark sync loop. Next run:", next_utc_minus_8_midnight())
        run_daily_loop(source=args.source)
        return 0

    result = scan_lark_updates(
        source=args.source,
        document_name=args.document_name,
        force=args.force,
        dry_run=args.dry_run,
        image_analysis_workers=args.image_workers,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
    return 0


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
