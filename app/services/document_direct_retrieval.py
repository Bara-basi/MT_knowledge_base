from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from psycopg import sql

from app.db.minio import (
    RawDocumentObject,
    build_minio_uri,
    get_minio_client,
    parse_raw_document_reference,
)
from app.db.postgres import postgres_connection
from app.schemas.retrieval import DocumentSearchMatch, FlowRetrievedChunk


TEXT_SUBDIRECTORY = "txt"
IMAGE_SUBDIRECTORY = "img"


@dataclass(frozen=True)
class ProcessedDocumentText:
    content: str
    text_objects: list[str]
    image_objects: list[str]


def search_processed_documents(
    query: str,
    *,
    limit: int = 10,
    table_name: str = "ingestion_registry",
) -> list[DocumentSearchMatch]:
    """Find processed document archive prefixes by registry document name.

    Exact document-name matches are returned first.  Fuzzy matches are used only
    when no exact match exists, so a highly specific filename cannot be drowned
    out by broad title-like matches.
    """

    cleaned = _clean_query(query)
    if not cleaned:
        return []

    exact = _select_registry_matches(
        cleaned,
        limit=limit,
        match_mode="exact",
        table_name=table_name,
    )
    if exact:
        return exact
    return _select_registry_matches(
        cleaned,
        limit=limit,
        match_mode="fuzzy",
        table_name=table_name,
    )


def build_direct_chunks(
    query: str,
    *,
    limit: int = 5,
    max_chars_per_document: int = 120_000,
) -> tuple[list[FlowRetrievedChunk], list[DocumentSearchMatch]]:
    matches = search_processed_documents(query, limit=limit)
    chunks: list[FlowRetrievedChunk] = []
    for index, match in enumerate(matches):
        processed = read_processed_document_text(
            match.processed_document_path,
            max_chars=max_chars_per_document,
        )
        if not processed.content.strip():
            continue
        chunks.append(
            FlowRetrievedChunk(
                chunk_id=f"document_direct:{_stable_key(match.processed_document_path)}",
                content=processed.content,
                chunk_index=0,
                chunk_type="document_direct",
                file_name=match.document_name,
                file_path=match.document_original_path or match.processed_document_path,
                path=match.processed_document_path,
                imgs=[
                    {
                        "index": image_index,
                        "img_name": PurePosixPath(object_name).name,
                        "img_path": build_minio_uri(
                            _parse_minio_uri(match.processed_document_path).bucket,
                            object_name,
                        ),
                    }
                    for image_index, object_name in enumerate(processed.image_objects)
                ]
                or None,
            )
        )
    return chunks, matches


def read_processed_document_text(
    processed_document_path: str,
    *,
    max_chars: int = 120_000,
) -> ProcessedDocumentText:
    reference = _parse_minio_uri(processed_document_path)
    prefix = reference.object_name.rstrip("/") + "/"
    client = get_minio_client()
    text_objects: list[str] = []
    image_objects: list[str] = []

    for item in client.list_objects(reference.bucket, prefix=prefix, recursive=True):
        object_name = str(getattr(item, "object_name", "") or "")
        if not object_name:
            continue
        relative_parts = PurePosixPath(object_name[len(prefix) :]).parts
        if TEXT_SUBDIRECTORY in relative_parts:
            text_objects.append(object_name)
        elif IMAGE_SUBDIRECTORY in relative_parts:
            image_objects.append(object_name)

    remaining = max_chars
    sections: list[str] = []
    for object_name in sorted(text_objects):
        if remaining <= 0:
            break
        text = _read_text_object(reference.bucket, object_name)
        if not text:
            continue
        if len(text) > remaining:
            text = text[:remaining] + "\n\n[文档直链内容已截断]"
        sections.append(f"### {PurePosixPath(object_name).name}\n{text}")
        remaining -= len(text)

    return ProcessedDocumentText(
        content="\n\n".join(sections),
        text_objects=sorted(text_objects),
        image_objects=sorted(image_objects),
    )


def _select_registry_matches(
    query: str,
    *,
    limit: int,
    match_mode: str,
    table_name: str,
) -> list[DocumentSearchMatch]:
    table = sql.Identifier(table_name)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            if match_mode == "exact":
                cur.execute(
                    sql.SQL(
                        """
                        SELECT document_name, document_original_path, processed_document_path
                        FROM {table}
                        WHERE processed_document_path IS NOT NULL
                          AND processed_document_path <> ''
                          AND (
                              document_name = %s
                              OR lower(document_name) = lower(%s)
                              OR lower(regexp_replace(document_name, '\\.[^.]*$', '')) = lower(%s)
                          )
                        ORDER BY document_name
                        LIMIT %s
                        """
                    ).format(table=table),
                    (query, query, _strip_extension(query), limit),
                )
            else:
                pattern = f"%{_escape_like(query)}%"
                stem_pattern = f"%{_escape_like(_strip_extension(query))}%"
                cur.execute(
                    sql.SQL(
                        """
                        SELECT document_name, document_original_path, processed_document_path
                        FROM {table}
                        WHERE processed_document_path IS NOT NULL
                          AND processed_document_path <> ''
                          AND (
                              document_name ILIKE %s ESCAPE '\\'
                              OR regexp_replace(document_name, '\\.[^.]*$', '') ILIKE %s ESCAPE '\\'
                          )
                        ORDER BY
                          CASE
                            WHEN document_name ILIKE %s ESCAPE '\\' THEN 0
                            ELSE 1
                          END,
                          length(document_name),
                          document_name
                        LIMIT %s
                        """
                    ).format(table=table),
                    (pattern, stem_pattern, pattern, limit),
                )
            rows = cur.fetchall()

    return [
        DocumentSearchMatch(
            document_name=str(row.get("document_name") or ""),
            document_original_path=str(row.get("document_original_path") or ""),
            processed_document_path=str(row.get("processed_document_path") or ""),
            match_mode=match_mode,  # type: ignore[arg-type]
        )
        for row in rows
    ]


def _parse_minio_uri(value: str) -> RawDocumentObject:
    parsed = urlparse(str(value).strip())
    if parsed.scheme not in {"minio", "s3"}:
        return parse_raw_document_reference(value)
    return RawDocumentObject(parsed.netloc, unquote(parsed.path).lstrip("/"))


def _read_text_object(bucket: str, object_name: str) -> str:
    response = get_minio_client().get_object(bucket, object_name)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _clean_query(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _strip_extension(value: str) -> str:
    path = PurePosixPath(str(value).strip())
    return path.stem if path.suffix else str(value).strip()


def _escape_like(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _stable_key(value: str) -> str:
    return (
        str(value)
        .lower()
        .replace("minio://", "")
        .replace("s3://", "")
        .replace("/", ":")
        .replace(" ", "_")
    )
