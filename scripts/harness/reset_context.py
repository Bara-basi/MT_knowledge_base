"""Delete Harness-only session state for a clean local test environment.

It deliberately does not touch knowledge documents, vector indexes, graph
data, or the normal encrypted chat_messages tables.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.db.minio import get_minio_client
from app.db.postgres import HARNESS_MEMORIES_TABLE, HARNESS_SESSIONS_TABLE, postgres_connection


def reset() -> tuple[int, int]:
    root = Path(settings.harness_session_root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    deleted_objects = 0
    client = get_minio_client()
    bucket = settings.harness_memory_bucket
    if client.bucket_exists(bucket):
        for item in client.list_objects(bucket, recursive=True):
            client.remove_object(bucket, item.object_name)
            deleted_objects += 1

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {HARNESS_MEMORIES_TABLE}")
            cur.execute(f"DELETE FROM {HARNESS_SESSIONS_TABLE}")
            deleted_rows = cur.rowcount or 0
    return deleted_rows, deleted_objects


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete only Harness session and memory state.")
    parser.add_argument("--confirm", action="store_true", help="Required acknowledgement for deletion.")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("Refusing to delete state without --confirm")
    rows, objects = reset()
    print(f"Reset complete: {rows} Harness sessions and {objects} memory objects deleted.")


if __name__ == "__main__":
    main()
