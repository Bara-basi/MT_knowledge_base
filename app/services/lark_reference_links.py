"""Resolve a retrieval file path to its canonical Feishu document URL."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from app.db.postgres import postgres_connection


def resolve_lark_document_link(reference_path: str) -> str | None:
    """Resolve a catalog path exactly before considering an unambiguous filename."""
    target = _normalize(reference_path)
    if not target:
        return None
    with postgres_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT document_name, document_link, path_titles, oss_object_key
            FROM lark_document_catalog
            WHERE document_link <> ''
            """
        )
        rows = cur.fetchall()
    exact: list[str] = []
    filename_matches: list[str] = []
    target_name = PurePosixPath(target).name.casefold()
    for row in rows:
        path = _normalize("/".join([*(row["path_titles"] or []), str(row["document_name"])]))
        key = _normalize(str(row["oss_object_key"] or ""))
        url = str(row["document_link"] or "").strip()
        if not url:
            continue
        if target in {path, key}:
            exact.append(url)
        if target_name and target_name == PurePosixPath(path).name.casefold():
            filename_matches.append(url)
    if len(set(exact)) == 1:
        return exact[0]
    unique_names = set(filename_matches)
    return next(iter(unique_names)) if len(unique_names) == 1 else None


def _normalize(value: str) -> str:
    return str(value or "").replace("\\", "/").strip().strip("/")
