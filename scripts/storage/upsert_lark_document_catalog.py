from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_file(PROJECT_ROOT / ".env")

from scripts.storage.lark_script_credentials import use_local_lark_credentials  # noqa: E402


use_local_lark_credentials()

from app.services.lark_document_catalog import (  # noqa: E402
    CATALOG_TABLE,
    DEFAULT_VECTOR_SRC,
    INGESTION_TABLE,
    catalog_summary,
    collect_records,
    ensure_catalog_table,
    sync_ingestion_registry_times,
    upsert_records,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect all configured Feishu/Lark knowledge documents into PostgreSQL.",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_VECTOR_SRC),
        help="Path to vector_src.json.",
    )
    parser.add_argument(
        "--catalog-table",
        default=CATALOG_TABLE,
        help=f"PostgreSQL catalog table name. Default: {CATALOG_TABLE}.",
    )
    parser.add_argument(
        "--ingestion-table",
        default=INGESTION_TABLE,
        help=f"PostgreSQL ingestion table name. Default: {INGESTION_TABLE}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect Feishu records and print counts without writing PostgreSQL.",
    )
    parser.add_argument(
        "--no-sync-ingestion-registry",
        action="store_true",
        help="Do not update ingestion_registry timestamps after catalog upsert.",
    )
    args = parser.parse_args()

    records, failures = collect_records(Path(args.source))
    result: dict[str, Any] = {
        "source": args.source,
        "catalog_table": args.catalog_table,
        "ingestion_table": args.ingestion_table,
        "candidate_rows": len(records),
        "failures": failures,
        "dry_run": args.dry_run,
    }

    if not args.dry_run:
        ensure_catalog_table(args.catalog_table)
        result["upserted_rows"] = upsert_records(records, args.catalog_table)
        if not args.no_sync_ingestion_registry:
            result["synced_ingestion_rows"] = sync_ingestion_registry_times(
                catalog_table_name=args.catalog_table,
                ingestion_table_name=args.ingestion_table,
            )
        result["catalog_summary"] = catalog_summary(args.catalog_table)

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
