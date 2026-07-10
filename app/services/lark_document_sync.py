from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse

from psycopg import sql

from app.db.minio import (
    DEFAULT_RAW_DOCUMENT_BUCKET,
    build_minio_uri,
    ensure_bucket,
    get_minio_client,
    guess_content_type,
    infer_prefix_from_name,
    parse_raw_document_reference,
)
from app.db.postgres import postgres_connection
from app.services.document_ingestion import ingest_document
from app.services.embedding import EmbeddingService
from app.services.lark_client import create_bitable_record, download_node, get_access_token
from app.services.lark_document_catalog import (
    CATALOG_TABLE,
    DEFAULT_VECTOR_SRC,
    INGESTION_TABLE,
    collect_link_record,
    collect_records,
    ensure_catalog_table,
    mark_missing_records_deleted,
    upsert_records,
)
from app.services.vector_store import VectorStoreService


PACIFIC_FIXED_TZ = timezone(timedelta(hours=-8))
NEW_DOCUMENT_BITABLE_APP_TOKEN = "Nbelb1BxwaJsDusKju4c8JUenih"
NEW_DOCUMENT_BITABLE_TABLE_ID = "tblmwcqbAIjP1FMg"
NEW_DOCUMENT_NAME_FIELD_ID = "fldzikCDJj"
NEW_DOCUMENT_CREATED_AT_FIELD_ID = "fldsWn62NY"
NEW_DOCUMENT_NAME_FIELD_NAME = "新增文档名"
NEW_DOCUMENT_CREATED_AT_FIELD_NAME = "新建时间"


@dataclass
class LarkSyncItemResult:
    document_name: str
    document_original_path: str
    document_link: str
    status: str
    reason: str = ""
    lark_updated_at: datetime | None = None
    registry_updated_at: datetime | None = None
    minio_sha256: str = ""
    lark_sha256: str = ""
    deleted_chunks: int = 0
    chunk_count: int = 0
    upsert_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_name": self.document_name,
            "document_original_path": self.document_original_path,
            "document_link": self.document_link,
            "status": self.status,
            "reason": self.reason,
            "lark_updated_at": self.lark_updated_at,
            "registry_updated_at": self.registry_updated_at,
            "minio_sha256": self.minio_sha256,
            "lark_sha256": self.lark_sha256,
            "deleted_chunks": self.deleted_chunks,
            "chunk_count": self.chunk_count,
            "upsert_count": self.upsert_count,
        }


@dataclass
class LarkSyncRunResult:
    source: str
    scanned_at: datetime
    catalog_rows: int
    catalog_failures: list[dict[str, Any]] = field(default_factory=list)
    candidate_count: int = 0
    new_document_count: int = 0
    deleted_document_count: int = 0
    checked_count: int = 0
    updated_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    items: list[LarkSyncItemResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "scanned_at": self.scanned_at,
            "catalog_rows": self.catalog_rows,
            "catalog_failures": self.catalog_failures,
            "candidate_count": self.candidate_count,
            "new_document_count": self.new_document_count,
            "deleted_document_count": self.deleted_document_count,
            "checked_count": self.checked_count,
            "updated_count": self.updated_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "items": [item.to_dict() for item in self.items],
        }


def scan_lark_updates(
    *,
    source: str | Path = DEFAULT_VECTOR_SRC,
    document_name: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    image_analysis_workers: int = 3,
    catalog_table: str = CATALOG_TABLE,
    ingestion_table: str = INGESTION_TABLE,
) -> LarkSyncRunResult:
    source_path = Path(source)
    scanned_at = datetime.now(timezone.utc)
    previous_catalog_keys = load_existing_catalog_keys(catalog_table)
    catalog_records, catalog_failures = collect_records(source_path)
    result = LarkSyncRunResult(
        source=str(source_path),
        scanned_at=scanned_at,
        catalog_rows=len(catalog_records),
        catalog_failures=catalog_failures,
    )

    access_token = get_access_token()
    vector_store_service = None if dry_run else VectorStoreService()

    new_records = find_new_catalog_records(catalog_records, previous_catalog_keys)
    result.new_document_count = len(new_records)
    for record in new_records:
        item = handle_new_catalog_record(record, access_token=access_token, dry_run=dry_run)
        result.items.append(item)
        if item.status == "failed":
            result.failed_count += 1

    # Deletion handling is intentionally disabled for now.
    # It is too risky to delete ingestion_registry rows and Milvus vectors based only on
    # a scan miss; re-enable this after adding an approval/soft-delete workflow.
    # deleted_registry_rows = load_deleted_registry_rows(
    #     catalog_records,
    #     ingestion_table=ingestion_table,
    #     document_name=document_name,
    # )
    # result.deleted_document_count = len(deleted_registry_rows)
    # for registry_row in deleted_registry_rows:
    #     item = handle_deleted_registry_row(
    #         registry_row,
    #         dry_run=dry_run,
    #         vector_store_service=vector_store_service,
    #         ingestion_table=ingestion_table,
    #     )
    #     result.items.append(item)
    #     if item.status == "deleted":
    #         result.updated_count += 1
    #     elif item.status == "failed":
    #         result.failed_count += 1

    if not dry_run:
        ensure_catalog_table(catalog_table)
        upsert_records(catalog_records, catalog_table)
        # Deletion marking is intentionally disabled for now. A missing item in one
        # scan should not mark catalog rows as deleted until we add confirmation.
        # mark_missing_records_deleted(
        #     [
        #         str(record.get("document_key"))
        #         for record in catalog_records
        #         if record.get("document_key")
        #     ],
        #     catalog_table,
        # )

    candidates = load_update_candidates(
        catalog_records,
        document_name=document_name,
        force=force,
        ingestion_table=ingestion_table,
    )
    result.candidate_count = len(candidates)
    if not candidates:
        return result

    embedding_service = None if dry_run else EmbeddingService()

    for candidate in candidates:
        item = sync_one_candidate(
            candidate,
            access_token=access_token,
            dry_run=dry_run,
            force=force,
            image_analysis_workers=image_analysis_workers,
            embedding_service=embedding_service,
            vector_store_service=vector_store_service,
        )
        result.items.append(item)
        result.checked_count += 1
        if item.status == "updated":
            result.updated_count += 1
            if not dry_run:
                sync_ingestion_registry_row_times(candidate, ingestion_table=ingestion_table)
        elif item.status == "failed":
            result.failed_count += 1
        else:
            result.skipped_count += 1
            if not dry_run and item.status == "skipped":
                sync_ingestion_registry_row_times(candidate, ingestion_table=ingestion_table)

    return result


def load_existing_catalog_keys(catalog_table: str = CATALOG_TABLE) -> set[str]:
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = current_schema()
                      AND table_name = %s
                ) AS exists
                """,
                (catalog_table,),
            )
            exists = bool((cur.fetchone() or {}).get("exists"))
            if not exists:
                return set()
            cur.execute(
                sql.SQL("SELECT document_key FROM {table} WHERE NOT is_deleted").format(
                    table=sql.Identifier(catalog_table)
                )
            )
            return {str(row["document_key"]) for row in cur.fetchall() if row.get("document_key")}


def find_new_catalog_records(
    catalog_records: list[dict[str, Any]],
    previous_catalog_keys: set[str],
) -> list[dict[str, Any]]:
    return [
        record
        for record in catalog_records
        if str(record.get("document_key") or "") not in previous_catalog_keys
    ]


def handle_new_catalog_record(
    catalog_record: dict[str, Any],
    *,
    access_token: str,
    dry_run: bool,
) -> LarkSyncItemResult:
    document_name = str(catalog_record.get("document_name") or "")
    output = LarkSyncItemResult(
        document_name=document_name,
        document_original_path="",
        document_link=str(catalog_record.get("document_link") or ""),
        status="new_document",
        reason="new Lark document found; approval record created",
        lark_updated_at=catalog_record.get("lark_updated_at"),
    )
    if dry_run:
        output.status = "would_create_approval_record"
        output.reason = "dry run"
        return output

    try:
        create_new_document_approval_record(catalog_record, access_token=access_token)
        return output
    except Exception as exc:
        output.status = "failed"
        output.reason = str(exc)
        return output


def create_new_document_approval_record(
    catalog_record: dict[str, Any],
    *,
    access_token: str,
) -> dict:
    created_at = ensure_aware_datetime(catalog_record.get("lark_created_at")) or datetime.now(timezone.utc)
    document_name = str(catalog_record.get("document_name") or "")
    document_link = str(catalog_record.get("document_link") or "")
    fields = {
        NEW_DOCUMENT_NAME_FIELD_NAME: {
            "text": document_name,
            "link": document_link,
        },
        NEW_DOCUMENT_CREATED_AT_FIELD_NAME: int(created_at.timestamp() * 1000),
    }
    return create_bitable_record(
        access_token,
        app_token=NEW_DOCUMENT_BITABLE_APP_TOKEN,
        table_id=NEW_DOCUMENT_BITABLE_TABLE_ID,
        fields=fields,
    )


def load_deleted_registry_rows(
    catalog_records: list[dict[str, Any]],
    *,
    ingestion_table: str,
    document_name: str | None,
) -> list[dict[str, Any]]:
    current_links = {
        normalize_document_link(str(record.get("document_link") or ""))
        for record in catalog_records
        if record.get("document_link")
    }
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT
                        id,
                        document_original_path,
                        document_name,
                        document_link,
                        created_at,
                        updated_at
                    FROM {table}
                    WHERE COALESCE(document_link, '') <> ''
                    """
                ).format(table=sql.Identifier(ingestion_table))
            )
            rows = [dict(row) for row in cur.fetchall()]

    if document_name:
        wanted_key = normalize_document_name(document_name)
        rows = [
            row
            for row in rows
            if normalize_document_name(str(row.get("document_name") or "")) == wanted_key
        ]

    return [
        row
        for row in rows
        if normalize_document_link(str(row.get("document_link") or "")) not in current_links
    ]


def handle_deleted_registry_row(
    registry_row: dict[str, Any],
    *,
    dry_run: bool,
    vector_store_service: VectorStoreService | None,
    ingestion_table: str,
) -> LarkSyncItemResult:
    document_name = str(registry_row.get("document_name") or "")
    output = LarkSyncItemResult(
        document_name=document_name,
        document_original_path=str(registry_row.get("document_original_path") or ""),
        document_link=str(registry_row.get("document_link") or ""),
        status="deleted",
        reason="registry document_link was not found in current Lark scan",
        registry_updated_at=registry_row.get("updated_at"),
    )
    if dry_run:
        output.status = "would_delete"
        output.reason = "dry run"
        return output
    try:
        if vector_store_service is None:
            raise RuntimeError("Delete requires vector store service")
        # Destructive deletion is intentionally disabled for now.
        # delete_result = vector_store_service.delete_by_metadata_document_name(
        #     document_name,
        #     metadata_keys=("file_path", "path"),
        #     flush=True,
        # )
        # output.deleted_chunks = int(delete_result.get("delete_count", 0) or 0)
        # delete_ingestion_registry_row(registry_row["id"], ingestion_table=ingestion_table)
        output.status = "delete_disabled"
        output.reason = "deletion is temporarily disabled"
        return output
    except Exception as exc:
        output.status = "failed"
        output.reason = str(exc)
        return output


def delete_ingestion_registry_row(row_id: Any, *, ingestion_table: str) -> None:
    # Destructive deletion is intentionally disabled for now.
    # with postgres_connection() as conn:
    #     with conn.cursor() as cur:
    #         cur.execute(
    #             sql.SQL("DELETE FROM {table} WHERE id = %s").format(
    #                 table=sql.Identifier(ingestion_table)
    #             ),
    #             (row_id,),
    #         )
    return None


def ingest_lark_document_link(
    document_link: str,
    *,
    bucket: str = DEFAULT_RAW_DOCUMENT_BUCKET,
    object_name: str | None = None,
    category: str | None = None,
    source_name: str | None = None,
    image_analysis_workers: int = 3,
    ingestion_table: str = INGESTION_TABLE,
) -> LarkSyncItemResult:
    access_token = get_access_token()
    catalog_record = collect_link_record(
        document_link,
        source_name=source_name,
        access_token=access_token,
    )
    document_name = str(catalog_record.get("document_name") or source_name or "document")
    document_link = str(catalog_record.get("document_link") or document_link)
    output = LarkSyncItemResult(
        document_name=document_name,
        document_original_path="",
        document_link=document_link,
        status="pending",
        lark_updated_at=catalog_record.get("lark_updated_at"),
    )

    try:
        with TemporaryDirectory(prefix="lark_ingest_") as temp_dir:
            temp_root = Path(temp_dir)
            lark_file = download_lark_current_file(
                access_token,
                catalog_record,
                temp_root / "lark",
            )
            target_object_name = object_name or default_minio_object_name_for_lark_document(
                document_name,
                category=category,
            )
            ensure_bucket(bucket)
            get_minio_client().fput_object(
                bucket,
                target_object_name,
                str(lark_file),
                content_type=guess_content_type(document_name),
            )
            document_original_path = build_minio_uri(bucket, target_object_name)
            output.document_original_path = document_original_path

            upsert_ingestion_registry_document(
                document_original_path=document_original_path,
                document_name=document_name,
                document_link=document_link,
                lark_created_at=catalog_record.get("lark_created_at"),
                lark_updated_at=catalog_record.get("lark_updated_at"),
                ingestion_table=ingestion_table,
            )

            embedding_service = EmbeddingService()
            vector_store_service = VectorStoreService()
            ingest_result = ingest_document(
                document_original_path,
                embedding_service=embedding_service,
                vector_store_service=vector_store_service,
                flush=True,
                image_analysis_workers=image_analysis_workers,
                rebuild=True,
                skip_existing_upsert=False,
            )
            output.chunk_count = ingest_result.chunk_count
            output.upsert_count = ingest_result.upsert_count
            output.status = "ingested"
            output.reason = "Lark document downloaded to MinIO and ingested"
            return output
    except Exception as exc:
        output.status = "failed"
        output.reason = str(exc)
        return output


def default_minio_object_name_for_lark_document(
    document_name: str,
    *,
    category: str | None = None,
) -> str:
    prefix = category or infer_prefix_from_name(document_name)
    return f"{prefix}/{PurePosixPath(document_name).name}"


def upsert_ingestion_registry_document(
    *,
    document_original_path: str,
    document_name: str,
    document_link: str,
    lark_created_at: Any,
    lark_updated_at: Any,
    ingestion_table: str,
) -> None:
    created_at = ensure_aware_datetime(lark_created_at)
    updated_at = ensure_aware_datetime(lark_updated_at) or created_at
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (
                        created_at,
                        updated_at,
                        document_original_path,
                        document_name,
                        document_link,
                        is_synced
                    )
                    VALUES (
                        COALESCE(%s, CURRENT_TIMESTAMP),
                        COALESCE(%s, CURRENT_TIMESTAMP),
                        %s,
                        %s,
                        %s,
                        TRUE
                    )
                    ON CONFLICT (document_original_path) DO UPDATE
                    SET created_at = COALESCE(EXCLUDED.created_at, {table}.created_at),
                        updated_at = COALESCE(EXCLUDED.updated_at, {table}.updated_at),
                        document_name = EXCLUDED.document_name,
                        document_link = EXCLUDED.document_link,
                        is_synced = TRUE
                    """
                ).format(table=sql.Identifier(ingestion_table)),
                (
                    created_at,
                    updated_at,
                    document_original_path,
                    document_name,
                    document_link,
                ),
            )


def load_update_candidates(
    catalog_records: list[dict[str, Any]],
    *,
    document_name: str | None,
    force: bool,
    ingestion_table: str,
) -> list[dict[str, Any]]:
    catalog_by_key = {
        normalize_document_name(record["document_name"]): record
        for record in catalog_records
        if record.get("document_name")
    }
    if document_name:
        wanted_key = normalize_document_name(document_name)
        catalog_by_key = {
            key: value
            for key, value in catalog_by_key.items()
            if key == wanted_key
        }
    if not catalog_by_key:
        return []

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                SELECT
                    id,
                    document_original_path,
                    document_name,
                    document_link,
                    created_at,
                    updated_at
                FROM {table}
                """
                ).format(table=sql.Identifier(ingestion_table))
            )
            registry_rows = [dict(row) for row in cur.fetchall()]

    candidates: list[dict[str, Any]] = []
    for registry_row in registry_rows:
        key = normalize_document_name(registry_row.get("document_name") or "")
        catalog_record = catalog_by_key.get(key)
        if catalog_record is None:
            continue

        lark_updated_at = catalog_record.get("lark_updated_at")
        registry_updated_at = registry_row.get("updated_at")
        if not force and not is_lark_newer(lark_updated_at, registry_updated_at):
            continue

        candidates.append(
            {
                "registry": registry_row,
                "catalog": catalog_record,
                "lark_updated_at": lark_updated_at,
                "registry_updated_at": registry_updated_at,
            }
        )

    return candidates


def sync_ingestion_registry_row_times(
    candidate: dict[str, Any],
    *,
    ingestion_table: str = INGESTION_TABLE,
) -> None:
    registry = candidate["registry"]
    catalog = candidate["catalog"]
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {table}
                    SET created_at = COALESCE(%s, created_at),
                        updated_at = COALESCE(%s, updated_at),
                        document_link = COALESCE(NULLIF(%s, ''), document_link),
                        is_synced = TRUE
                    WHERE id = %s
                    """
                ).format(table=sql.Identifier(ingestion_table)),
                (
                    catalog.get("lark_created_at"),
                    catalog.get("lark_updated_at") or catalog.get("lark_created_at"),
                    catalog.get("document_link") or "",
                    registry.get("id"),
                ),
            )


def sync_one_candidate(
    candidate: dict[str, Any],
    *,
    access_token: str,
    dry_run: bool,
    force: bool,
    image_analysis_workers: int,
    embedding_service: EmbeddingService | None,
    vector_store_service: VectorStoreService | None,
) -> LarkSyncItemResult:
    registry = candidate["registry"]
    catalog = candidate["catalog"]
    document_name = str(registry.get("document_name") or catalog.get("document_name") or "")
    document_original_path = str(registry.get("document_original_path") or "")
    output = LarkSyncItemResult(
        document_name=document_name,
        document_original_path=document_original_path,
        document_link=str(catalog.get("document_link") or registry.get("document_link") or ""),
        status="pending",
        lark_updated_at=candidate.get("lark_updated_at"),
        registry_updated_at=candidate.get("registry_updated_at"),
    )

    try:
        reference = parse_raw_document_reference(document_original_path)
        with TemporaryDirectory(prefix="lark_sync_") as temp_dir:
            temp_root = Path(temp_dir)
            minio_file = temp_root / "minio" / PurePosixPath(reference.object_name).name
            minio_file.parent.mkdir(parents=True, exist_ok=True)
            get_minio_client().fget_object(reference.bucket, reference.object_name, str(minio_file))

            lark_file = download_lark_current_file(
                access_token,
                catalog,
                temp_root / "lark",
            )

            output.minio_sha256 = file_sha256(minio_file)
            output.lark_sha256 = file_sha256(lark_file)
            if output.minio_sha256 == output.lark_sha256 and not force:
                output.status = "skipped"
                output.reason = "content hash unchanged"
                return output

            if dry_run:
                output.status = "would_update"
                output.reason = "dry run"
                return output

            upload_file_to_existing_minio_object(lark_file, reference)
            if vector_store_service is None or embedding_service is None:
                raise RuntimeError("Sync requires embedding and vector store services")

            # Destructive metadata-name deletion is intentionally disabled for now.
            # The following ingest upsert still handles normal file_id replacement.
            # delete_result = vector_store_service.delete_by_metadata_document_name(
            #     document_name,
            #     metadata_keys=("file_path", "path"),
            #     flush=True,
            # )
            # output.deleted_chunks = int(delete_result.get("delete_count", 0) or 0)

            ingest_result = ingest_document(
                document_original_path,
                embedding_service=embedding_service,
                vector_store_service=vector_store_service,
                flush=True,
                image_analysis_workers=image_analysis_workers,
                rebuild=True,
                skip_existing_upsert=False,
            )
            output.chunk_count = ingest_result.chunk_count
            output.upsert_count = ingest_result.upsert_count
            output.status = "updated"
            return output
    except Exception as exc:
        output.status = "failed"
        output.reason = str(exc)
        return output


def download_lark_current_file(
    access_token: str,
    catalog_record: dict[str, Any],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_node = dict(catalog_record.get("raw_node") or {})
    if not raw_node:
        raise ValueError(f"Catalog row is missing raw_node: {catalog_record.get('document_name')}")

    before = {path.resolve() for path in output_dir.rglob("*") if path.is_file()}
    processed, result = download_node(
        access_token,
        raw_node,
        output_dir,
        used_paths=set(),
        overwrite=True,
    )
    if not processed:
        raise RuntimeError(result)

    result_path = Path(str(result))
    if result_path.exists():
        return result_path

    after = [path for path in output_dir.rglob("*") if path.is_file() and path.resolve() not in before]
    if not after:
        raise FileNotFoundError(f"Lark download did not create a file: {result}")
    return max(after, key=lambda path: path.stat().st_mtime)


def upload_file_to_existing_minio_object(path: Path, reference) -> None:
    client = get_minio_client()
    client.fput_object(
        reference.bucket,
        reference.object_name,
        str(path),
        content_type=guess_content_type(PurePosixPath(reference.object_name).name),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_lark_newer(lark_updated_at: Any, registry_updated_at: Any) -> bool:
    lark_dt = ensure_aware_datetime(lark_updated_at)
    registry_dt = ensure_aware_datetime(registry_updated_at)
    if lark_dt is None:
        return False
    if registry_dt is None:
        return True
    return lark_dt > registry_dt


def ensure_aware_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def normalize_document_name(value: str) -> str:
    return "".join(str(value or "").lower().split())


def normalize_document_link(value: str) -> str:
    parsed = urlparse(str(value or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return str(value or "").strip().rstrip("/")
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"


def next_utc_minus_8_midnight(now: datetime | None = None) -> datetime:
    current = now or datetime.now(timezone.utc)
    current_fixed = current.astimezone(PACIFIC_FIXED_TZ)
    next_day = current_fixed.date() + timedelta(days=1)
    next_run_fixed = datetime.combine(next_day, datetime.min.time(), tzinfo=PACIFIC_FIXED_TZ)
    return next_run_fixed.astimezone(timezone.utc)


def run_daily_loop(
    *,
    source: str | Path = DEFAULT_VECTOR_SRC,
    stop_after_runs: int | None = None,
    poll_seconds: float = 60.0,
) -> None:
    completed = 0
    while stop_after_runs is None or completed < stop_after_runs:
        next_run = next_utc_minus_8_midnight()
        while True:
            seconds = (next_run - datetime.now(timezone.utc)).total_seconds()
            if seconds <= 0:
                break
            time.sleep(min(max(poll_seconds, 1.0), seconds))

        scan_lark_updates(source=source)
        completed += 1
