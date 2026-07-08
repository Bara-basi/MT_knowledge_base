from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines before importing app settings."""

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_file(PROJECT_ROOT / ".env")

from psycopg import sql  # noqa: E402

from app.db.minio import parse_raw_document_reference  # noqa: E402
from app.db.postgres import postgres_connection  # noqa: E402
from scripts.ingestion.parse_documents import find_document_files  # noqa: E402


TABLE_NAME = "ingestion_registry"
TABLE_COMMENT = "\u5165\u5e93\u8868"
MAPPING_DIR = PROJECT_ROOT / "data" / "metadata" / "local2lark_mapping"


def normalize_filename_key(filename: str) -> str:
    return re.sub(r"\s+", "", filename)


def load_lark_link_mapping(mapping_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(mapping_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Mapping file must contain a JSON object: {path}")
        for filename, link in data.items():
            if not isinstance(filename, str):
                continue
            normalized = normalize_filename_key(filename)
            if normalized and normalized not in mapping:
                mapping[normalized] = str(link or "")
    return mapping


def split_version_original_stem(object_name: str) -> str | None:
    parts = PurePosixPath(object_name.replace("\\", "/")).parts
    for part in parts[:-1]:
        split_suffix = "(\u5207\u5206\u7248)"
        if part.endswith(split_suffix):
            return part[: -len(split_suffix)]
    return None


def lark_link_for_object(object_name: str, mapping: dict[str, str]) -> str:
    filename = PurePosixPath(object_name).name
    direct_link = mapping.get(normalize_filename_key(filename))
    if direct_link:
        return direct_link

    original_stem = split_version_original_stem(object_name)
    if original_stem:
        normalized_stem = normalize_filename_key(original_stem)
        for filename_key, link in mapping.items():
            if PurePosixPath(filename_key).stem == normalized_stem and link:
                return link
    return ""


def build_registry_rows(mapping: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_uri in find_document_files("", recursive=True):
        reference = parse_raw_document_reference(file_uri)
        document_name = PurePosixPath(reference.object_name).name
        document_link = lark_link_for_object(reference.object_name, mapping)
        rows.append(
            {
                "document_original_path": reference.uri,
                "document_name": document_name,
                "document_link": document_link,
                "is_synced": bool(document_link),
            }
        )
    return rows


def ensure_ingestion_registry_table(table_name: str = TABLE_NAME) -> None:
    table = sql.Identifier(table_name)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        document_original_path TEXT NOT NULL UNIQUE,
                        document_name TEXT NOT NULL,
                        document_link TEXT NOT NULL DEFAULT '',
                        is_synced BOOLEAN NOT NULL DEFAULT FALSE
                    )
                    """
                ).format(table=table)
            )
            cur.execute(
                sql.SQL("COMMENT ON TABLE {table} IS {comment}").format(
                    table=table,
                    comment=sql.Literal(TABLE_COMMENT),
                )
            )
            column_comments = {
                "created_at": "\u5165\u5e93\u65e5\u671f",
                "updated_at": "\u66f4\u65b0\u65e5\u671f",
                "document_original_path": "\u6587\u6863\u539f\u8def\u5f84",
                "document_name": "\u6587\u6863\u540d",
                "document_link": "\u6587\u6863\u94fe\u63a5\uff08\u98de\u4e66\u94fe\u63a5\uff09",
                "is_synced": "\u662f\u5426\u540c\u6b65",
            }
            for column, comment in column_comments.items():
                cur.execute(
                    sql.SQL("COMMENT ON COLUMN {table}.{column} IS {comment}").format(
                        table=table,
                        column=sql.Identifier(column),
                        comment=sql.Literal(comment),
                    )
                )


def upsert_registry_rows(rows: list[dict[str, Any]], table_name: str = TABLE_NAME) -> int:
    table = sql.Identifier(table_name)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {table} (
                        document_original_path,
                        document_name,
                        document_link,
                        is_synced
                    )
                    VALUES (
                        %(document_original_path)s,
                        %(document_name)s,
                        %(document_link)s,
                        %(is_synced)s
                    )
                    ON CONFLICT (document_original_path) DO UPDATE
                    SET updated_at = CURRENT_TIMESTAMP,
                        document_name = EXCLUDED.document_name,
                        document_link = EXCLUDED.document_link,
                        is_synced = EXCLUDED.is_synced
                    """
                ).format(table=table),
                rows,
            )
    return len(rows)


def table_summary(table_name: str = TABLE_NAME) -> dict[str, Any]:
    table = sql.Identifier(table_name)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE is_synced) AS synced,
                        COUNT(*) FILTER (WHERE NOT is_synced) AS not_synced,
                        MIN(created_at) AS first_created_at,
                        MAX(updated_at) AS last_updated_at
                    FROM {table}
                    """
                ).format(table=table)
            )
            row = cur.fetchone()
    return dict(row or {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and populate the document ingestion registry table.",
    )
    parser.add_argument(
        "--table-name",
        default=TABLE_NAME,
        help=f"PostgreSQL table name. Default: {TABLE_NAME}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build rows and print summary without writing PostgreSQL.",
    )
    args = parser.parse_args()

    mapping = load_lark_link_mapping(MAPPING_DIR)
    rows = build_registry_rows(mapping)

    result: dict[str, Any] = {
        "table_name": args.table_name,
        "mapping_count": len(mapping),
        "candidate_rows": len(rows),
        "candidate_synced": sum(1 for row in rows if row["is_synced"]),
        "candidate_not_synced": sum(1 for row in rows if not row["is_synced"]),
        "dry_run": args.dry_run,
    }

    if not args.dry_run:
        ensure_ingestion_registry_table(args.table_name)
        result["upserted_rows"] = upsert_registry_rows(rows, args.table_name)
        result["table_summary"] = table_summary(args.table_name)

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
