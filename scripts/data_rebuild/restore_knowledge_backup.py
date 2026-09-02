"""Restore a backup produced by backup_and_clear_knowledge.py.

The script refuses to overwrite non-empty knowledge targets.  This makes the
backup a recovery point, not a merge/import tool.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.host", override=True)

from psycopg.types.json import Jsonb  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db.minio import ensure_bucket, get_minio_client, guess_content_type  # noqa: E402
from app.db.milvus import ensure_chunk_collection, get_milvus_client  # noqa: E402
from app.db.neo4j import get_neo4j_driver, validate_neo4j_identifier  # noqa: E402
from app.db.postgres import postgres_connection  # noqa: E402


LOCAL_PATHS = (Path("data/raw"), Path("data/processing"), Path("data/metadata/local2lark_mapping"))
POSTGRES_TABLES = ("ingestion_registry", "lark_document_catalog", "marketing_asset_catalog")


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def require_empty_targets(backup_root: Path, manifest: dict[str, Any]) -> None:
    for relative in LOCAL_PATHS:
        target = PROJECT_ROOT / relative
        if target.exists() and any(target.iterdir()):
            raise RuntimeError(f"Refusing to overwrite non-empty local path: {target}")
    client = get_minio_client()
    for bucket in manifest["scope"]["minio_buckets"]:
        if any(not item.is_dir for item in client.list_objects(bucket, recursive=True)):
            raise RuntimeError(f"Refusing to overwrite non-empty MinIO bucket: {bucket}")
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            for table in POSTGRES_TABLES:
                cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
                if int(cur.fetchone()["count"]):
                    raise RuntimeError(f"Refusing to overwrite non-empty PostgreSQL table: {table}")
    client = get_milvus_client(timeout=60)
    collection = manifest["scope"]["milvus_collection"]
    if client.has_collection(collection) and int(client.get_collection_stats(collection).get("row_count", 0)):
        raise RuntimeError(f"Refusing to overwrite non-empty Milvus collection: {collection}")
    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.neo4j_database) as session:
            count = int((session.run("MATCH (n) RETURN count(n) AS count").single() or {}).get("count", 0))
            if count:
                raise RuntimeError("Refusing to overwrite non-empty Neo4j database")
    finally:
        driver.close()


def restore_local(backup_root: Path) -> None:
    for relative in LOCAL_PATHS:
        source = backup_root / "local" / relative
        target = PROJECT_ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.copytree(source, target, copy_function=shutil.copy2, dirs_exist_ok=True)


def restore_minio(backup_root: Path, buckets: list[str]) -> None:
    client = get_minio_client()
    for bucket in buckets:
        ensure_bucket(bucket)
        for item in json.loads((backup_root / "minio" / f"{bucket}.objects.json").read_text(encoding="utf-8")):
            source = backup_root / "minio" / bucket / Path(*str(item["object_name"]).split("/"))
            if not source.is_file() or source.stat().st_size != int(item["size"]):
                raise RuntimeError(f"Invalid MinIO backup file: {source}")
            client.fput_object(bucket, str(item["object_name"]), str(source), content_type=guess_content_type(source.name))


def restore_postgres(backup_root: Path) -> None:
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            for table in POSTGRES_TABLES:
                payload = read_gzip_json(backup_root / "postgres" / f"{table}.json.gz")
                columns = [str(column["column_name"]) for column in payload["columns"]]
                json_columns = {str(column["column_name"]) for column in payload["columns"] if column["data_type"] == "jsonb"}
                placeholders = ", ".join(["%s"] * len(columns))
                statement = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
                for row in payload["rows"]:
                    values = [Jsonb(row[column]) if column in json_columns and row[column] is not None else row[column] for column in columns]
                    cur.execute(statement, values)
                cur.execute(f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), COALESCE((SELECT MAX(id) FROM {table}), 1), true)")


def normalize_sparse_vector(row: dict[str, Any]) -> dict[str, Any]:
    sparse = row.get("sparse_vector")
    if isinstance(sparse, dict):
        row["sparse_vector"] = {int(key): value for key, value in sparse.items()}
    return row


def chunks(values: list[dict[str, Any]], size: int = 256) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def restore_milvus(backup_root: Path, collection: str) -> int:
    source = backup_root / "milvus" / f"{collection}.jsonl.gz"
    rows = [normalize_sparse_vector(row) for row in jsonl_rows(source)]
    client = get_milvus_client(timeout=60)
    if client.has_collection(collection):
        client.drop_collection(collection)
    ensure_chunk_collection(client)
    for batch in chunks(rows):
        client.upsert(collection_name=collection, data=batch)
    client.flush(collection_name=collection)
    actual = int(client.get_collection_stats(collection).get("row_count", -1))
    if actual != len(rows):
        raise RuntimeError(f"Milvus restore count mismatch: {actual} != {len(rows)}")
    return actual


def restore_neo4j(backup_root: Path) -> dict[str, int]:
    payload = read_gzip_json(backup_root / "neo4j" / "graph.json.gz")
    driver = get_neo4j_driver()
    nodes = list(payload["nodes"])
    relationships = list(payload["relationships"])
    try:
        with driver.session(database=settings.neo4j_database) as session:
            for node in nodes:
                properties = dict(node["properties"])
                node_id = str(properties["id"])
                labels = [validate_neo4j_identifier(label) for label in node["labels"]]
                session.run("MERGE (n:GraphNode {id: $id}) SET n += $properties", id=node_id, properties=properties).consume()
                for label in labels:
                    if label != "GraphNode":
                        session.run(f"MATCH (n:GraphNode {{id: $id}}) SET n:{label}", id=node_id).consume()
            for relationship in relationships:
                rel_type = validate_neo4j_identifier(str(relationship["type"]))
                properties = dict(relationship["properties"])
                session.run(
                    f"MATCH (source:GraphNode {{id: $source_id}}), (target:GraphNode {{id: $target_id}}) "
                    f"MERGE (source)-[r:{rel_type} {{id: $id}}]->(target) SET r += $properties",
                    source_id=str(relationship["source_id"]),
                    target_id=str(relationship["target_id"]),
                    id=str(properties["id"]),
                    properties=properties,
                ).consume()
            restored_nodes = int((session.run("MATCH (n) RETURN count(n) AS count").single() or {}).get("count", 0))
            restored_relationships = int((session.run("MATCH ()-[r]->() RETURN count(r) AS count").single() or {}).get("count", 0))
            if restored_nodes != len(nodes) or restored_relationships != len(relationships):
                raise RuntimeError("Neo4j restore count mismatch")
            return {"nodes": restored_nodes, "relationships": restored_relationships}
    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup_dir", type=Path)
    args = parser.parse_args()
    backup_root = args.backup_dir.resolve()
    if not (backup_root / "READY_TO_RESTORE.json").is_file():
        raise SystemExit(f"Backup is not marked restorable: {backup_root}")
    manifest = json.loads((backup_root / "manifest.json").read_text(encoding="utf-8"))
    require_empty_targets(backup_root, manifest)
    restore_local(backup_root)
    restore_minio(backup_root, list(manifest["scope"]["minio_buckets"]))
    restore_postgres(backup_root)
    milvus_rows = restore_milvus(backup_root, str(manifest["scope"]["milvus_collection"]))
    neo4j_result = restore_neo4j(backup_root)
    print(json.dumps({"restored_from": str(backup_root), "milvus_rows": milvus_rows, "neo4j": neo4j_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
