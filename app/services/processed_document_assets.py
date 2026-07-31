"""Archive parsed text and images without moving local ingestion artifacts.

The object prefix intentionally contains the *full* raw filename (including its
extension).  Local legacy processing folders are named by stem, which makes
``guide.docx`` and ``guide.pdf`` collide.  Including the extension in MinIO
keeps the durable archive and registry unambiguous.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from psycopg import sql

from app.core.config import settings
from app.db.minio import (
    build_minio_uri,
    ensure_bucket,
    get_minio_client,
    guess_content_type,
    parse_raw_document_reference,
)
from app.db.postgres import postgres_connection
from app.services.parser.paths import processing_document_dir


PROCESSED_DOCUMENT_BUCKET_DEFAULT = "knowledge-processed-docs"
ARCHIVED_SUBDIRECTORIES = frozenset({"txt", "img", "json"})


def processed_document_bucket() -> str:
    return settings.minio_processed_document_bucket or PROCESSED_DOCUMENT_BUCKET_DEFAULT


def processed_document_prefix(source_file: str | Path) -> str:
    """Return the stable archive prefix for one raw document.

    For ``category/guide.docx`` this is ``category/guide.docx``.  Keeping the
    extension is deliberate: it prevents same-stem documents from overwriting
    one another in object storage.
    """

    reference = parse_raw_document_reference(source_file)
    return reference.object_name


def processed_document_uri(source_file: str | Path, *, bucket: str | None = None) -> str:
    return build_minio_uri(bucket or processed_document_bucket(), processed_document_prefix(source_file))


def synchronize_processed_assets(
    source_file: str | Path,
    *,
    produced_processing_dir: str | Path | None = None,
    update_registry: bool = True,
) -> str:
    """Mirror ``txt``, ``img`` and parser JSON output to MinIO, then record it.

    Parser downloads can live under a transient MinIO cache.  Their generated
    files are first copied to the canonical local processing path derived from
    the original source, so downstream chunk/embedding code sees the same path.
    Existing local chunk and embedding files are never touched.
    """

    canonical_dir = processing_document_dir(source_file)
    produced_dir = Path(produced_processing_dir or canonical_dir)
    _copy_asset_subtrees(produced_dir, canonical_dir)
    if not any(
        path.is_file() and "txt" in path.relative_to(canonical_dir).parts
        for path in canonical_dir.rglob("*")
    ):
        raise RuntimeError(f"Parser produced no txt assets for: {source_file}")
    _upload_asset_subtrees(canonical_dir, source_file)
    uri = processed_document_uri(source_file)
    if update_registry:
        set_registry_processed_document_path(source_file, uri)
    return uri


def _copy_asset_subtrees(source_dir: Path, destination_dir: Path) -> int:
    if not source_dir.exists():
        raise FileNotFoundError(f"Parser output directory does not exist: {source_dir}")
    copied = 0
    for source in _asset_files(source_dir):
        relative = source.relative_to(source_dir)
        destination = destination_dir / relative
        if source.resolve() == destination.resolve():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    return copied


def _upload_asset_subtrees(processing_dir: Path, source_file: str | Path) -> int:
    if not processing_dir.exists():
        raise FileNotFoundError(f"Canonical processing directory does not exist: {processing_dir}")
    bucket = ensure_bucket(processed_document_bucket())
    prefix = processed_document_prefix(source_file).strip("/")
    client = get_minio_client()
    uploaded = 0
    for asset in _asset_files(processing_dir):
        relative = asset.relative_to(processing_dir).as_posix()
        client.fput_object(
            bucket,
            f"{prefix}/{relative}",
            str(asset),
            content_type=guess_content_type(asset.name),
        )
        uploaded += 1
    return uploaded


def _asset_files(processing_dir: Path):
    for path in processing_dir.rglob("*"):
        if path.is_file() and any(part in ARCHIVED_SUBDIRECTORIES for part in path.relative_to(processing_dir).parts):
            yield path


def set_registry_processed_document_path(
    source_file: str | Path,
    processed_path: str,
    *,
    table_name: str = "ingestion_registry",
) -> None:
    """Set the archive prefix for an existing registry row.

    A missing row is an integrity error: silently succeeding would leave an
    uploaded archive that cannot be discovered by source-document Q&A.
    """

    table = sql.Identifier(table_name)
    source_uri = parse_raw_document_reference(source_file).uri
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS processed_document_path TEXT").format(table=table)
            )
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {table}
                    SET processed_document_path = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE document_original_path = %s
                    """
                ).format(table=table),
                (processed_path, source_uri),
            )
            if cur.rowcount != 1:
                raise LookupError(f"No ingestion registry row for raw document: {source_uri}")
