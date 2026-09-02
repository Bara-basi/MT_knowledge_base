"""Clear the old knowledge-base data using a completed, restorable backup."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_rebuild.backup_and_clear_knowledge import (  # noqa: E402
    clear_local_paths,
    clear_minio_bucket,
    clear_milvus,
    clear_neo4j,
    clear_postgres,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup_dir", type=Path, help="A backup directory containing READY_TO_RESTORE.json")
    args = parser.parse_args()
    backup_root = args.backup_dir.resolve()
    ready = backup_root / "READY_TO_RESTORE.json"
    manifest_path = backup_root / "manifest.json"
    if not ready.is_file() or not manifest_path.is_file():
        raise SystemExit(f"Refusing to clear without a completed backup: {backup_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    buckets = list(manifest["scope"]["minio_buckets"])
    result = {
        "cleared_at": datetime.now(timezone.utc).isoformat(),
        "backup_dir": str(backup_root),
        "local": "cleared",
        "minio": {},
    }
    clear_local_paths()
    for bucket in buckets:
        result["minio"][bucket] = clear_minio_bucket(bucket)
    result["postgres"] = clear_postgres()
    result["milvus"] = clear_milvus(str(manifest["scope"]["milvus_collection"]))
    result["neo4j"] = clear_neo4j()
    (backup_root / "clear-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
