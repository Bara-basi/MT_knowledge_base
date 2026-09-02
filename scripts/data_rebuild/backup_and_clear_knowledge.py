"""Back up and clear the old knowledge-base data set.

The destructive phase is available only with ``--apply`` and starts only after
every backup source has been copied and recorded in the manifest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.host", override=True)

from app.core.config import settings  # noqa: E402
from app.db.minio import ensure_bucket, get_minio_client  # noqa: E402
from app.db.milvus import MilvusCollectionConfig, get_milvus_client  # noqa: E402
from app.db.neo4j import get_neo4j_driver  # noqa: E402
from app.db.postgres import postgres_connection  # noqa: E402
from minio.deleteobjects import DeleteObject  # noqa: E402


LOCAL_PATHS = (
    Path("data/raw"),
    Path("data/processing"),
    Path("data/metadata/local2lark_mapping"),
)
POSTGRES_TABLES = (
    "ingestion_registry",
    "lark_document_catalog",
    "marketing_asset_catalog",
)


def json_default(value: Any) -> str:
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_summary(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
    return {
        "exists": root.exists(),
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }


def copy_local_paths(backup_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relative in LOCAL_PATHS:
        source = PROJECT_ROOT / relative
        destination = backup_root / "local" / relative
        if source.exists():
            shutil.copytree(source, destination, copy_function=shutil.copy2)
        source_summary = tree_summary(source)
        destination_summary = tree_summary(destination)
        if source_summary != destination_summary:
            raise RuntimeError(f"Local backup verification failed for {relative}: {source_summary} != {destination_summary}")
        result[relative.as_posix()] = {"source": source_summary, "backup": destination_summary}
    return result


def backup_minio_bucket(bucket: str, backup_root: Path) -> dict[str, Any]:
    client = get_minio_client()
    destination = backup_root / "minio" / bucket
    objects: list[dict[str, Any]] = []
    for item in client.list_objects(bucket, recursive=True):
        if item.is_dir:
            continue
        object_name = str(item.object_name)
        output = destination.joinpath(*object_name.split("/"))
        output.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(bucket, object_name, str(output))
        actual_size = output.stat().st_size
        expected_size = int(item.size or 0)
        if actual_size != expected_size:
            raise RuntimeError(f"MinIO backup size mismatch: {bucket}/{object_name}")
        objects.append(
            {
                "object_name": object_name,
                "size": expected_size,
                "etag": str(item.etag or ""),
                "last_modified": str(item.last_modified or ""),
                "sha256": sha256(output),
            }
        )
    metadata_path = backup_root / "minio" / f"{bucket}.objects.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(objects, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"objects": len(objects), "bytes": sum(item["size"] for item in objects), "metadata": str(metadata_path.relative_to(backup_root))}


def write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, default=json_default)


def backup_postgres(backup_root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            for table in POSTGRES_TABLES:
                cur.execute(f"SELECT * FROM {table} ORDER BY id")
                rows = [dict(row) for row in cur.fetchall()]
                cur.execute(
                    "SELECT column_name, data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position",
                    (table,),
                )
                columns = [dict(row) for row in cur.fetchall()]
                path = backup_root / "postgres" / f"{table}.json.gz"
                write_gzip_json(path, {"table": table, "columns": columns, "rows": rows})
                output[table] = {"rows": len(rows), "file": str(path.relative_to(backup_root))}
    return output


def backup_milvus(backup_root: Path, collection: str) -> dict[str, Any]:
    client = get_milvus_client(timeout=60)
    if not client.has_collection(collection):
        return {"exists": False, "rows": 0}
    client.load_collection(collection)
    description = client.describe_collection(collection)
    rows_path = backup_root / "milvus" / f"{collection}.jsonl.gz"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    iterator = client.query_iterator(
        collection_name=collection,
        filter='id != ""',
        output_fields=["id", "vector", "sparse_vector", "content", "file_id", "chunk_index", "metadata"],
        # Keep response and serialization sizes bounded.  A large batch can
        # appear stalled when every row contains a dense embedding vector.
        batch_size=128,
    )
    try:
        with gzip.open(rows_path, "wt", encoding="utf-8") as handle:
            for batch in iter(iterator.next, []):
                for row in batch:
                    handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
                    count += 1
                handle.flush()
                if count % 1024 == 0:
                    print(f"Milvus backup progress: {count} rows", flush=True)
    finally:
        iterator.close()
    write_gzip_json(backup_root / "milvus" / f"{collection}.schema.json.gz", description)
    stats = client.get_collection_stats(collection)
    if int(stats.get("row_count", -1)) != count:
        raise RuntimeError(f"Milvus backup count mismatch for {collection}: {stats} vs {count}")
    return {"exists": True, "rows": count, "stats": stats, "rows_file": str(rows_path.relative_to(backup_root))}


def backup_neo4j(backup_root: Path) -> dict[str, Any]:
    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.neo4j_database) as session:
            nodes = [
                dict(row)
                for row in session.run("MATCH (n) RETURN labels(n) AS labels, properties(n) AS properties ORDER BY n.id")
            ]
            relationships = [
                dict(row)
                for row in session.run(
                    "MATCH (source)-[r]->(target) "
                    "RETURN source.id AS source_id, target.id AS target_id, type(r) AS type, properties(r) AS properties "
                    "ORDER BY r.id"
                )
            ]
    finally:
        driver.close()
    path = backup_root / "neo4j" / "graph.json.gz"
    write_gzip_json(path, {"nodes": nodes, "relationships": relationships})
    return {"nodes": len(nodes), "relationships": len(relationships), "file": str(path.relative_to(backup_root))}


def clear_local_paths() -> None:
    for relative in LOCAL_PATHS:
        path = PROJECT_ROOT / relative
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def clear_minio_bucket(bucket: str) -> int:
    client = get_minio_client()
    names = [str(item.object_name) for item in client.list_objects(bucket, recursive=True) if not item.is_dir]
    if names:
        errors = list(client.remove_objects(bucket, [DeleteObject(name) for name in names]))
        if errors:
            raise RuntimeError(f"MinIO deletion failed for {bucket}: {errors[:3]}")
    remaining = [item.object_name for item in client.list_objects(bucket, recursive=True) if not item.is_dir]
    if remaining:
        raise RuntimeError(f"MinIO bucket was not emptied: {bucket}")
    return len(names)


def clear_postgres() -> dict[str, int]:
    counts: dict[str, int] = {}
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            for table in POSTGRES_TABLES:
                cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
                counts[table] = int(cur.fetchone()["count"])
                cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY")
    return counts


def clear_milvus(collection: str) -> dict[str, Any]:
    client = get_milvus_client(timeout=60)
    if not client.has_collection(collection):
        return {"existed": False, "rows_deleted": 0}
    rows = int(client.get_collection_stats(collection).get("row_count", 0))
    client.drop_collection(collection)
    if client.has_collection(collection):
        raise RuntimeError(f"Milvus collection still exists: {collection}")
    return {"existed": True, "rows_deleted": rows}


def clear_neo4j() -> dict[str, int]:
    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.neo4j_database) as session:
            result = session.run("MATCH (n) DETACH DELETE n").consume()
            counters = result.counters
            remaining = session.run("MATCH (n) RETURN count(n) AS count").single()
            if int((remaining or {}).get("count", -1)) != 0:
                raise RuntimeError("Neo4j graph was not emptied")
            return {"nodes_deleted": counters.nodes_deleted, "relationships_deleted": counters.relationships_deleted}
    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Perform the clear after a verified backup.")
    parser.add_argument("--backup-dir", type=Path, help="Backup output directory. Defaults under data/backup.")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("knowledge-reset-%Y%m%dT%H%M%SZ")
    backup_root = args.backup_dir or PROJECT_ROOT / "data" / "backup" / run_id
    backup_root = backup_root.resolve()
    if backup_root.exists() and any(backup_root.iterdir()):
        raise SystemExit(f"Refusing to use a non-empty backup directory: {backup_root}")
    backup_root.mkdir(parents=True, exist_ok=True)
    collection = MilvusCollectionConfig().name
    buckets = (settings.minio_bucket, settings.minio_processed_document_bucket)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "apply": args.apply,
        "scope": {"local_paths": [path.as_posix() for path in LOCAL_PATHS], "minio_buckets": buckets, "postgres_tables": POSTGRES_TABLES, "milvus_collection": collection, "neo4j_database": settings.neo4j_database},
    }
    manifest["local"] = copy_local_paths(backup_root)
    manifest["minio"] = {bucket: backup_minio_bucket(bucket, backup_root) for bucket in buckets}
    manifest["postgres"] = backup_postgres(backup_root)
    manifest["milvus"] = backup_milvus(backup_root, collection)
    manifest["neo4j"] = backup_neo4j(backup_root)
    (backup_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    (backup_root / "READY_TO_RESTORE.json").write_text(json.dumps({"manifest": "manifest.json", "ready_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    print(json.dumps({"backup_dir": str(backup_root), "backup": manifest}, ensure_ascii=False, indent=2, default=json_default))
    if not args.apply:
        print("Dry run complete: backup created; no knowledge data was deleted.")
        return

    clear_result = {
        "local": "cleared",
        "minio": {bucket: clear_minio_bucket(bucket) for bucket in buckets},
        "postgres": clear_postgres(),
        "milvus": clear_milvus(collection),
        "neo4j": clear_neo4j(),
    }
    (backup_root / "clear-result.json").write_text(json.dumps(clear_result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cleared": clear_result, "backup_dir": str(backup_root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
