from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.milvus import MilvusCollectionConfig, get_milvus_client


def main() -> None:
    _configure_utf8_stdio()

    parser = argparse.ArgumentParser(
        description="Delete Milvus chunks whose metadata path field matches a value or prefix.",
    )
    parser.add_argument(
        "path",
        help='Path field value or prefix, for example "data\\processing\\产品标准".',
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Delete only exact matches. By default the value is treated as a prefix.",
    )
    parser.add_argument(
        "--metadata-key",
        default="path",
        help="Metadata field to match. Default: path.",
    )
    parser.add_argument(
        "--collection",
        help="Milvus collection name. Default comes from app settings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Milvus query page size. Default: 4096.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show matching chunk ids without deleting.",
    )
    parser.add_argument(
        "--no-flush",
        action="store_true",
        help="Skip flushing after deletion.",
    )
    args = parser.parse_args()

    config = MilvusCollectionConfig()
    collection_name = args.collection or config.name
    client = get_milvus_client(timeout=60.0)
    if not client.has_collection(collection_name):
        raise SystemExit(f"Milvus collection does not exist: {collection_name}")

    client.load_collection(collection_name)
    matched_ids = find_matching_chunk_ids(
        client,
        collection_name=collection_name,
        metadata_key=args.metadata_key,
        path_value=args.path,
        exact=args.exact,
        batch_size=args.batch_size,
    )

    print(f"Collection: {collection_name}")
    print(f"Metadata key: {args.metadata_key}")
    print(f"Mode: {'exact' if args.exact else 'prefix'}")
    print(f"Matched chunks: {len(matched_ids)}")
    if matched_ids:
        print(f"First id: {matched_ids[0]}")
        print(f"Last id: {matched_ids[-1]}")

    if args.dry_run or not matched_ids:
        print("Deleted chunks: 0")
        return

    delete_count = delete_chunk_ids(
        client,
        collection_name=collection_name,
        ids=matched_ids,
        batch_size=args.batch_size,
    )
    if not args.no_flush:
        client.flush(collection_name=collection_name)
    print(f"Deleted chunks: {delete_count}")


def find_matching_chunk_ids(
    client,
    *,
    collection_name: str,
    metadata_key: str,
    path_value: str,
    exact: bool,
    batch_size: int,
) -> list[str]:
    target = _normalize_path(path_value)
    matched_ids: list[str] = []
    query_filter = 'id != ""'
    iterator = client.query_iterator(
        collection_name=collection_name,
        filter=query_filter,
        output_fields=["id", "metadata"],
        batch_size=batch_size,
    )

    try:
        while True:
            rows = iterator.next()
            if not rows:
                break

            for row in rows:
                metadata = row.get("metadata") or {}
                value = _normalize_path(str(metadata.get(metadata_key) or ""))
                if _matches(value, target, exact=exact):
                    matched_ids.append(str(row["id"]))
    finally:
        iterator.close()

    return matched_ids


def delete_chunk_ids(
    client,
    *,
    collection_name: str,
    ids: list[str],
    batch_size: int,
) -> int:
    deleted = 0
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        result = client.delete(collection_name=collection_name, ids=batch)
        deleted += int(_result_value(result, "delete_count", len(batch)) or 0)
    return deleted


def _matches(value: str, target: str, *, exact: bool) -> bool:
    if exact:
        return value == target
    return value.startswith(target)


def _normalize_path(value: str) -> str:
    return str(value or "").strip().replace("\\", "/").rstrip("/")


def _result_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
