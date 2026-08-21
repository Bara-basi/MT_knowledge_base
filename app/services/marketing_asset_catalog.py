"""Persistent, non-RAG catalogue and keyword search for marketing assets."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from psycopg import sql

from app.db.postgres import postgres_connection


TABLE_NAME = "marketing_asset_catalog"
_TOKEN_SPLIT = re.compile(r"[\s/\\,;|]+")


@dataclass(frozen=True)
class MarketingAssetMatch:
    library_name: str
    full_path: str
    document_name: str
    feishu_link: str
    score: int
    match_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "library_name": self.library_name,
            "full_path": self.full_path,
            "document_name": self.document_name,
            "feishu_link": self.feishu_link,
            "score": self.score,
            "match_mode": self.match_mode,
        }


def ensure_marketing_asset_catalog(table_name: str = TABLE_NAME) -> None:
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
                        library_name TEXT NOT NULL,
                        full_path TEXT NOT NULL,
                        document_name TEXT NOT NULL,
                        feishu_link TEXT NOT NULL UNIQUE,
                        normalized_path TEXT NOT NULL,
                        source_file TEXT NOT NULL,
                        last_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN NOT NULL DEFAULT TRUE
                    )
                    """
                ).format(table=table)
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {index} ON {table} (is_active)").format(
                    index=sql.Identifier(f"{table_name}_active_idx"), table=table
                )
            )


def normalize_search_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def prepare_asset_row(
    *,
    library_name: str,
    path: str,
    feishu_link: str,
    source_file: str,
) -> dict[str, str]:
    clean_library = str(library_name).strip()
    clean_path = str(path).strip().strip("/")
    full_path = "/".join(part for part in (clean_library, clean_path) if part)
    if not full_path or not str(feishu_link).strip():
        raise ValueError("marketing asset path and feishu_link are required")
    return {
        "library_name": clean_library,
        "full_path": full_path,
        "document_name": clean_path.rsplit("/", 1)[-1],
        "feishu_link": str(feishu_link).strip(),
        "normalized_path": normalize_search_text(full_path),
        "source_file": str(source_file).strip(),
    }


def upsert_marketing_assets(
    rows: Iterable[dict[str, str]], *, table_name: str = TABLE_NAME
) -> int:
    values = list(rows)
    if not values:
        return 0
    table = sql.Identifier(table_name)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {table} (
                        library_name, full_path, document_name, feishu_link,
                        normalized_path, source_file, last_seen_at, is_active
                    ) VALUES (
                        %(library_name)s, %(full_path)s, %(document_name)s, %(feishu_link)s,
                        %(normalized_path)s, %(source_file)s, CURRENT_TIMESTAMP, TRUE
                    )
                    ON CONFLICT (feishu_link) DO UPDATE
                    SET updated_at = CURRENT_TIMESTAMP,
                        library_name = EXCLUDED.library_name,
                        full_path = EXCLUDED.full_path,
                        document_name = EXCLUDED.document_name,
                        normalized_path = EXCLUDED.normalized_path,
                        source_file = EXCLUDED.source_file,
                        last_seen_at = CURRENT_TIMESTAMP,
                        is_active = TRUE
                    """
                ).format(table=table),
                values,
            )
    return len(values)


def search_marketing_assets(
    query: str, *, limit: int = 10, table_name: str = TABLE_NAME
) -> list[MarketingAssetMatch]:
    normalized_query = normalize_search_text(query)
    if not normalized_query:
        return []
    tokens = [normalize_search_text(token) for token in _TOKEN_SPLIT.split(str(query))]
    tokens = [token for token in tokens if token]
    table = sql.Identifier(table_name)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT library_name, full_path, document_name, feishu_link, normalized_path
                    FROM {table}
                    WHERE is_active
                      AND normalized_path LIKE %s ESCAPE '\\'
                    """
                ).format(table=table),
                (f"%{_escape_like(normalized_query)}%",),
            )
            rows = cur.fetchall()

    matches = [_score_match(row, normalized_query, tokens) for row in rows]
    matches.sort(key=lambda item: (-item.score, len(item.full_path), item.full_path))
    return matches[:limit]


def format_harness_results(query: str, matches: list[MarketingAssetMatch]) -> str:
    if not matches:
        return f"未找到与“{query}”匹配的营销资料。"
    lines = [f"找到 {len(matches)} 条与“{query}”匹配的营销资料："]
    for index, match in enumerate(matches, start=1):
        lines.extend(
            (
                f"{index}. 路径：{match.full_path}",
                f"   飞书链接：{match.feishu_link}",
            )
        )
    return "\n".join(lines)


def _score_match(row: dict[str, Any], query: str, tokens: list[str]) -> MarketingAssetMatch:
    title = normalize_search_text(str(row.get("document_name") or ""))
    path = str(row.get("normalized_path") or "")
    if title == query:
        score, mode = 100, "exact_name"
    elif title.startswith(query):
        score, mode = 90, "name_prefix"
    elif query in title:
        score, mode = 80, "name_contains"
    else:
        score, mode = 60, "path_contains"
    score += min(20, sum(1 for token in tokens if token in path) * 4)
    return MarketingAssetMatch(
        library_name=str(row.get("library_name") or ""),
        full_path=str(row.get("full_path") or ""),
        document_name=str(row.get("document_name") or ""),
        feishu_link=str(row.get("feishu_link") or ""),
        score=score,
        match_mode=mode,
    )


def _escape_like(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
