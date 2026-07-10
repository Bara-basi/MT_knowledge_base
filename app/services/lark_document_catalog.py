from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psycopg import sql
from psycopg.types.json import Jsonb

from app.db.postgres import postgres_connection
from app.services.lark_client import (
    build_wiki_view_url,
    collect_wiki_root_nodes,
    get_access_token,
    get_docx_document,
    get_export_ext,
    get_node,
    list_child_nodes,
    load_vector_sources,
    node_title,
    parse_feishu_url,
)


CATALOG_TABLE = "lark_document_catalog"
INGESTION_TABLE = "ingestion_registry"
DEFAULT_VECTOR_SRC = Path("data") / "src" / "vector_src.json"


def is_supported_document_node(node: dict[str, Any]) -> bool:
    obj_type = (node.get("obj_type") or "").lower()
    return obj_type == "file" or get_export_ext(obj_type) is not None


def sanitize_file_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" .")
    return name or "untitled"


def document_name_for_node(node: dict[str, Any]) -> str:
    title = sanitize_file_name(node_title(node))
    obj_type = (node.get("obj_type") or "").lower()
    if obj_type == "file":
        return title

    ext = get_export_ext(obj_type)
    if not ext:
        return title

    suffix = Path(title).suffix.lstrip(".")
    if suffix.lower() == ext.lower():
        return title
    return f"{title}.{ext}"


def parse_lark_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            value = int(stripped)
        else:
            try:
                parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
            except ValueError:
                return None
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 100_000_000_000:
            timestamp = timestamp / 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return None


def first_timestamp(node: dict[str, Any], keys: tuple[str, ...]) -> datetime | None:
    for key in keys:
        parsed = parse_lark_timestamp(node.get(key))
        if parsed is not None:
            return parsed
    return None


def lark_created_at(node: dict[str, Any]) -> datetime | None:
    return first_timestamp(
        node,
        ("obj_create_time", "create_time", "created_time", "node_create_time", "created_at"),
    )


def lark_updated_at(node: dict[str, Any]) -> datetime | None:
    return first_timestamp(
        node,
        (
            "obj_edit_time",
            "edit_time",
            "update_time",
            "updated_time",
            "modified_time",
            "node_edit_time",
            "updated_at",
        ),
    )


def document_key_for_node(node: dict[str, Any], document_link: str) -> str:
    node_token = str(node.get("node_token") or "").strip()
    if node_token:
        return f"wiki:{node_token}"

    obj_type = str(node.get("obj_type") or "").strip()
    obj_token = str(node.get("obj_token") or "").strip()
    if obj_type and obj_token:
        return f"obj:{obj_type}:{obj_token}"
    return f"link:{document_link}"


def build_record(
    *,
    source_type: str,
    source_name: str,
    source_url: str,
    node: dict[str, Any],
    path_titles: list[str],
) -> dict[str, Any] | None:
    if not is_supported_document_node(node):
        return None

    document_link = build_wiki_view_url(node, source_url)
    created_at = lark_created_at(node)
    updated_at = lark_updated_at(node) or created_at
    return {
        "document_key": document_key_for_node(node, document_link),
        "source_type": source_type,
        "source_name": source_name,
        "document_name": document_name_for_node(node),
        "document_title": node_title(node),
        "document_link": document_link,
        "lark_created_at": created_at,
        "lark_updated_at": updated_at,
        "obj_type": str(node.get("obj_type") or ""),
        "obj_token": str(node.get("obj_token") or ""),
        "node_token": str(node.get("node_token") or ""),
        "space_id": str(node.get("space_id") or ""),
        "parent_node_token": str(node.get("parent_node_token") or ""),
        "path_titles": path_titles,
        "raw_node": node,
    }


def merge_record(records: dict[str, dict[str, Any]], record: dict[str, Any] | None) -> None:
    if record is not None:
        records.setdefault(record["document_key"], record)


def collect_wiki_node_records(
    access_token: str,
    *,
    node: dict[str, Any],
    source_name: str,
    source_url: str,
    records: dict[str, dict[str, Any]],
    failures: list[dict[str, Any]],
    path_titles: list[str],
) -> None:
    title = node_title(node)
    current_path = path_titles + [title]
    merge_record(
        records,
        build_record(
            source_type="wiki",
            source_name=source_name,
            source_url=source_url,
            node=node,
            path_titles=current_path,
        ),
    )
    if not node.get("has_child"):
        return

    try:
        children = list(list_child_nodes(access_token, node["space_id"], node["node_token"]))
    except Exception as exc:
        failures.append(
            {
                "source_type": "wiki",
                "source_name": source_name,
                "source_url": source_url,
                "title": title,
                "node_token": node.get("node_token", ""),
                "stage": "list_child_nodes",
                "reason": str(exc),
            }
        )
        return

    for child in children:
        collect_wiki_node_records(
            access_token,
            node=child,
            source_name=source_name,
            source_url=source_url,
            records=records,
            failures=failures,
            path_titles=current_path,
        )


def collect_wiki_source_records(
    access_token: str,
    *,
    source_name: str,
    source_url: str,
    records: dict[str, dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    link_type, token = parse_feishu_url(source_url)
    if link_type != "wiki" or not token:
        raise ValueError(f"wiki source must be a wiki link: {source_url}")

    root_node = get_node(access_token, token)
    root_node.setdefault("url", source_url)
    try:
        root_nodes = collect_wiki_root_nodes(access_token, root_node)
    except Exception as exc:
        failures.append(
            {
                "source_type": "wiki",
                "source_name": source_name,
                "source_url": source_url,
                "title": node_title(root_node),
                "node_token": root_node.get("node_token", ""),
                "stage": "detect_space_root",
                "reason": str(exc),
            }
        )
        root_nodes = None

    nodes = root_nodes if root_nodes is not None else [root_node]
    for node in nodes:
        collect_wiki_node_records(
            access_token,
            node=node,
            source_name=source_name,
            source_url=source_url,
            records=records,
            failures=failures,
            path_titles=[source_name],
        )


def collect_direct_link_record(
    access_token: str,
    *,
    source_name: str,
    source_url: str,
    records: dict[str, dict[str, Any]],
) -> None:
    link_type, token = parse_feishu_url(source_url)
    if not token:
        raise ValueError(f"unsupported Feishu link: {source_url}")

    if link_type == "wiki":
        node = get_node(access_token, token)
        node.setdefault("url", source_url)
    elif link_type == "docx":
        document = get_docx_document(access_token, token)
        node = {
            "title": document.get("title") or source_name,
            "obj_type": "docx",
            "obj_token": token,
            "node_token": token,
            "url": source_url,
            "create_time": document.get("create_time"),
            "update_time": document.get("update_time"),
            "raw_document": document,
        }
    else:
        obj_type = {
            "docs": "doc",
            "doc": "doc",
            "sheets": "sheet",
            "base": "bitable",
            "bitable": "bitable",
            "slides": "slides",
            "file": "file",
        }.get(link_type)
        if not obj_type:
            raise ValueError(f"unsupported Feishu link type={link_type}: {source_url}")
        node = {"title": source_name, "obj_type": obj_type, "obj_token": token, "node_token": token, "url": source_url}

    merge_record(
        records,
        build_record(
            source_type="single_file",
            source_name=source_name,
            source_url=source_url,
            node=node,
            path_titles=[source_name],
        ),
    )


def collect_records(source_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = load_vector_sources(source_path)
    access_token = get_access_token()
    records: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for source_name, link in sources["single_file"].items():
        try:
            collect_direct_link_record(access_token, source_name=source_name, source_url=link, records=records)
        except Exception as exc:
            failures.append({"source_type": "single_file", "source_name": source_name, "source_url": link, "stage": "resolve_source", "reason": str(exc)})

    for source_name, link in sources["wiki"].items():
        try:
            collect_wiki_source_records(access_token, source_name=source_name, source_url=link, records=records, failures=failures)
        except Exception as exc:
            failures.append({"source_type": "wiki", "source_name": source_name, "source_url": link, "stage": "resolve_source", "reason": str(exc)})

    return list(records.values()), failures


def collect_link_record(
    document_link: str,
    *,
    source_name: str | None = None,
    access_token: str | None = None,
) -> dict[str, Any]:
    token = access_token or get_access_token()
    records: dict[str, dict[str, Any]] = {}
    collect_direct_link_record(
        token,
        source_name=source_name or document_link,
        source_url=document_link,
        records=records,
    )
    if not records:
        raise ValueError(f"No supported document found for Lark link: {document_link}")
    return next(iter(records.values()))


def ensure_catalog_table(table_name: str = CATALOG_TABLE) -> None:
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
                        raw_node JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        last_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        is_deleted BOOLEAN NOT NULL DEFAULT FALSE
                    )
                    """
                ).format(table=table)
            )
            for suffix, column in (
                ("document_name_idx", "document_name"),
                ("document_link_idx", "document_link"),
                ("lark_updated_at_idx", "lark_updated_at"),
            ):
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {index} ON {table} ({column})").format(
                        index=sql.Identifier(f"{table_name}_{suffix}"),
                        table=table,
                        column=sql.Identifier(column),
                    )
                )


def upsert_records(rows: list[dict[str, Any]], table_name: str = CATALOG_TABLE) -> int:
    if not rows:
        return 0
    table = sql.Identifier(table_name)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {table} (
                        document_key, source_type, source_name, document_name,
                        document_title, document_link, lark_created_at, lark_updated_at,
                        obj_type, obj_token, node_token, space_id, parent_node_token,
                        path_titles, raw_node, last_seen_at, is_deleted
                    )
                    VALUES (
                        %(document_key)s, %(source_type)s, %(source_name)s,
                        %(document_name)s, %(document_title)s, %(document_link)s,
                        %(lark_created_at)s, %(lark_updated_at)s, %(obj_type)s,
                        %(obj_token)s, %(node_token)s, %(space_id)s,
                        %(parent_node_token)s, %(path_titles)s, %(raw_node)s,
                        CURRENT_TIMESTAMP, FALSE
                    )
                    ON CONFLICT (document_key) DO UPDATE
                    SET updated_at = CURRENT_TIMESTAMP,
                        source_type = EXCLUDED.source_type,
                        source_name = EXCLUDED.source_name,
                        document_name = EXCLUDED.document_name,
                        document_title = EXCLUDED.document_title,
                        document_link = EXCLUDED.document_link,
                        lark_created_at = EXCLUDED.lark_created_at,
                        lark_updated_at = EXCLUDED.lark_updated_at,
                        obj_type = EXCLUDED.obj_type,
                        obj_token = EXCLUDED.obj_token,
                        node_token = EXCLUDED.node_token,
                        space_id = EXCLUDED.space_id,
                        parent_node_token = EXCLUDED.parent_node_token,
                        path_titles = EXCLUDED.path_titles,
                        raw_node = EXCLUDED.raw_node,
                        last_seen_at = CURRENT_TIMESTAMP,
                        is_deleted = FALSE
                    """
                ).format(table=table),
                [{**row, "path_titles": Jsonb(row["path_titles"]), "raw_node": Jsonb(row["raw_node"])} for row in rows],
            )
    return len(rows)


def mark_missing_records_deleted(
    document_keys: list[str],
    table_name: str = CATALOG_TABLE,
) -> int:
    table = sql.Identifier(table_name)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            if document_keys:
                cur.execute(
                    sql.SQL(
                        """
                        UPDATE {table}
                        SET is_deleted = TRUE,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE NOT is_deleted
                          AND NOT (document_key = ANY(%s))
                        """
                    ).format(table=table),
                    (document_keys,),
                )
            else:
                cur.execute(
                    sql.SQL(
                        """
                        UPDATE {table}
                        SET is_deleted = TRUE,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE NOT is_deleted
                        """
                    ).format(table=table)
                )
            return cur.rowcount or 0


def sync_ingestion_registry_times(
    *,
    catalog_table_name: str = CATALOG_TABLE,
    ingestion_table_name: str = INGESTION_TABLE,
) -> int:
    catalog_table = sql.Identifier(catalog_table_name)
    ingestion_table = sql.Identifier(ingestion_table_name)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    WITH catalog_dedup AS (
                        SELECT
                            regexp_replace(lower(document_name), '\\s+', '', 'g') AS document_name_key,
                            MIN(lark_created_at) AS lark_created_at,
                            MAX(COALESCE(lark_updated_at, lark_created_at)) AS lark_updated_at
                        FROM {catalog_table}
                        WHERE NOT is_deleted AND document_name <> ''
                        GROUP BY regexp_replace(lower(document_name), '\\s+', '', 'g')
                    )
                    UPDATE {ingestion_table} AS registry
                    SET created_at = COALESCE(catalog_dedup.lark_created_at, registry.created_at),
                        updated_at = COALESCE(catalog_dedup.lark_updated_at, registry.updated_at)
                    FROM catalog_dedup
                    WHERE regexp_replace(lower(registry.document_name), '\\s+', '', 'g') = catalog_dedup.document_name_key
                      AND (
                          registry.created_at IS DISTINCT FROM COALESCE(catalog_dedup.lark_created_at, registry.created_at)
                          OR registry.updated_at IS DISTINCT FROM COALESCE(catalog_dedup.lark_updated_at, registry.updated_at)
                      )
                    """
                ).format(catalog_table=catalog_table, ingestion_table=ingestion_table)
            )
            return cur.rowcount or 0


def catalog_summary(table_name: str = CATALOG_TABLE) -> dict[str, Any]:
    table = sql.Identifier(table_name)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE NOT is_deleted) AS active,
                        COUNT(*) FILTER (WHERE lark_created_at IS NULL) AS missing_created_at,
                        COUNT(*) FILTER (WHERE lark_updated_at IS NULL) AS missing_updated_at,
                        MIN(lark_created_at) AS earliest_lark_created_at,
                        MAX(lark_updated_at) AS latest_lark_updated_at
                    FROM {table}
                    """
                ).format(table=table)
            )
            row = cur.fetchone()
    return dict(row or {})
