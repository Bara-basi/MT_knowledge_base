from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.milvus import check_milvus_health, drop_chunk_collection, ensure_chunk_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize Milvus for MTSCO KB.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check Milvus connectivity; do not create or load collections.",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop the existing chunk collection before initializing it.",
    )
    args = parser.parse_args()

    if args.check_only:
        result = check_milvus_health()
    else:
        if args.drop_existing:
            drop_chunk_collection()
        result = ensure_chunk_collection()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
