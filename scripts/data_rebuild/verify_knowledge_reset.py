"""Verify that the knowledge-base targets are empty or match a backup manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.host", override=True)

from app.core.config import settings  # noqa: E402
from app.db.minio import get_minio_client  # noqa: E402
from app.db.milvus import MilvusCollectionConfig, get_milvus_client  # noqa: E402
from app.db.neo4j import get_neo4j_driver  # noqa: E402
from app.db.postgres import postgres_connection  # noqa: E402


LOCAL_PATHS = (Path("data/raw"), Path("data/processing"), Path("data/metadata/local2lark_mapping"))
POSTGRES_TABLES = ("ingestion_registry", "lark_document_catalog", "marketing_asset_catalog")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-restored", type=Path, help="Compare live row/object counts against this backup manifest.")
    args = parser.parse_args()
    client = get_minio_client()
    buckets = (settings.minio_bucket, settings.minio_processed_document_bucket)
    minio = {bucket: sum(1 for item in client.list_objects(bucket, recursive=True) if not item.is_dir) for bucket in buckets}
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            postgres = {}
            for table in POSTGRES_TABLES:
                cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
                postgres[table] = int(cur.fetchone()["count"])
    milvus_client = get_milvus_client(timeout=60)
    collection = MilvusCollectionConfig().name
    milvus = int(milvus_client.get_collection_stats(collection).get("row_count", 0)) if milvus_client.has_collection(collection) else 0
    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.neo4j_database) as session:
            neo4j = {"nodes": int((session.run("MATCH (n) RETURN count(n) AS count").single() or {}).get("count", 0)), "relationships": int((session.run("MATCH ()-[r]->() RETURN count(r) AS count").single() or {}).get("count", 0))}
    finally:
        driver.close()
    local = {path.as_posix(): sum(1 for item in (PROJECT_ROOT / path).rglob("*") if item.is_file()) if (PROJECT_ROOT / path).exists() else 0 for path in LOCAL_PATHS}
    result = {"local_files": local, "minio_objects": minio, "postgres_rows": postgres, "milvus_rows": milvus, "neo4j": neo4j}
    if args.expect_restored:
        manifest = json.loads((args.expect_restored.resolve() / "manifest.json").read_text(encoding="utf-8"))
        expected = {"minio_objects": {bucket: manifest["minio"][bucket]["objects"] for bucket in buckets}, "postgres_rows": {table: manifest["postgres"][table]["rows"] for table in POSTGRES_TABLES}, "milvus_rows": manifest["milvus"]["rows"], "neo4j": {"nodes": manifest["neo4j"]["nodes"], "relationships": manifest["neo4j"]["relationships"]}}
        result["expected"] = expected
        if any(result[key] != expected[key] for key in expected):
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit("Restore verification failed")
    else:
        non_empty = any(local.values()) or any(minio.values()) or any(postgres.values()) or milvus or any(neo4j.values())
        result["is_empty"] = not non_empty
        if non_empty:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit("Knowledge reset verification failed: data remains")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
