"""Copy the local Lark document catalog to a forwarded PostgreSQL target."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from psycopg.types.json import Jsonb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.host", override=True)

import psycopg  # noqa: E402

from app.db.postgres import postgres_connection  # noqa: E402


COLUMNS = (
    "document_key", "source_type", "source_name", "document_name", "document_title",
    "document_link", "lark_created_at", "lark_updated_at", "obj_type", "obj_token",
    "node_token", "space_id", "parent_node_token", "path_titles", "raw_node",
    "oss_object_key", "oss_uri", "content_sha256", "content_size", "ingested_at", "last_seen_at",
)


def target_connection(args):
    password = args.password or os.getenv("TARGET_POSTGRES_PASSWORD", "")
    if not password:
        raise RuntimeError("Set TARGET_POSTGRES_PASSWORD or pass --password")
    return psycopg.connect(host=args.host, port=args.port, dbname=args.database, user=args.user, password=password)


def rebuild_target_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS lark_document_catalog")
        cur.execute(
            """
            CREATE TABLE lark_document_catalog (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                document_key TEXT NOT NULL UNIQUE,
                source_type TEXT NOT NULL,
                source_name TEXT NOT NULL,
                document_name TEXT NOT NULL,
                document_title TEXT NOT NULL DEFAULT '',
                document_link TEXT NOT NULL DEFAULT '',
                lark_created_at TIMESTAMPTZ,
                lark_updated_at TIMESTAMPTZ,
                obj_type TEXT NOT NULL DEFAULT '',
                obj_token TEXT NOT NULL DEFAULT '',
                node_token TEXT NOT NULL DEFAULT '',
                space_id TEXT NOT NULL DEFAULT '',
                parent_node_token TEXT NOT NULL DEFAULT '',
                path_titles JSONB NOT NULL DEFAULT '[]'::jsonb,
                raw_node JSONB NOT NULL DEFAULT '{}'::jsonb,
                oss_object_key TEXT,
                oss_uri TEXT,
                content_sha256 TEXT,
                content_size BIGINT,
                ingested_at TIMESTAMPTZ,
                last_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute("CREATE INDEX lark_document_catalog_document_name_idx ON lark_document_catalog (document_name)")
        cur.execute("CREATE INDEX lark_document_catalog_document_link_idx ON lark_document_catalog (document_link)")
        cur.execute("CREATE INDEX lark_document_catalog_lark_updated_at_idx ON lark_document_catalog (lark_updated_at)")


def source_rows() -> list[dict]:
    with postgres_connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(COLUMNS)} FROM lark_document_catalog ORDER BY document_key")
        return list(cur.fetchall())


def migrate(conn, rows: list[dict]) -> None:
    placeholders = ", ".join(f"%({name})s" for name in COLUMNS)
    updates = ", ".join(f"{name} = EXCLUDED.{name}" for name in COLUMNS if name != "document_key")
    statement = f"INSERT INTO lark_document_catalog ({', '.join(COLUMNS)}) VALUES ({placeholders}) ON CONFLICT (document_key) DO UPDATE SET {updates}"
    payload = []
    for row in rows:
        item = dict(row)
        item["path_titles"] = Jsonb(item["path_titles"] or [])
        item["raw_node"] = Jsonb(item["raw_node"] or {})
        payload.append(item)
    with conn.cursor() as cur:
        cur.executemany(statement, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--database", default="mtsco_knowledge_base")
    parser.add_argument("--user", default="mtsco")
    parser.add_argument("--password", default="")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rebuild-table", action="store_true", help="Drop and recreate the target table before importing.")
    args = parser.parse_args()
    rows = source_rows()
    with target_connection(args) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.lark_document_catalog')")
            has_target_table = cur.fetchone()[0] is not None
            if has_target_table:
                cur.execute("SELECT count(*) FROM lark_document_catalog")
                target_count = cur.fetchone()[0]
            else:
                target_count = 0
        print({"source_rows": len(rows), "target_rows_before": target_count, "apply": args.apply}, flush=True)
        if not args.apply:
            return 0
        if not args.rebuild_table:
            raise RuntimeError("Refusing to modify target without --rebuild-table")
        rebuild_target_table(conn)
        migrate(conn, rows)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM lark_document_catalog")
            print({"target_rows_after": cur.fetchone()[0]}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
