"""Parse OSS-backed Lark knowledge, embed it, and upsert it into Milvus.

Only parser TXT/image assets live in ``$HARNESS_WORKDIR/knowledge``.  OSS source
downloads, parser work files, chunks and embeddings live temporarily in
``data/processing``; chunk and embedding JSON are removed after vector writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Production FastAPI is a host systemd service: its unit loads .env.host after
# .env to replace Docker service DNS names with localhost endpoints.  Mirror
# that order for this manually-run rebuild command.
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.host", override=True)

from app.core.config import settings  # noqa: E402
from app.db.postgres import postgres_connection  # noqa: E402
from app.services.lark_client import sanitize_path_part  # noqa: E402
from app.services.chunking.splitter import build_file_id, save_chunks, split_items  # noqa: E402
from app.services.embedding import (  # noqa: E402
    EmbeddingService,
    default_bm25_model_file,
    load_bm25_embedding_function,
    load_chunks,
)
from app.services.knowledge_quality import is_acceptable_knowledge_file  # noqa: E402
from app.services.parser.parser import parse_document  # noqa: E402
from app.services.parser.paths import processing_document_dir  # noqa: E402
from app.services.vector_store import VectorStoreService  # noqa: E402

SUPPORTED_SUFFIXES = {".docx", ".xlsx", ".pptx", ".pdf"}
def _harness_knowledge_root() -> Path:
    configured = Path(settings.harness_workdir).expanduser()
    workdir = configured if configured.is_absolute() else PROJECT_ROOT / configured
    return workdir.resolve() / "knowledge"


HARNESS_KNOWLEDGE_ROOT = _harness_knowledge_root()
PROCESSING_ROOT = PROJECT_ROOT / "data" / "processing"
DEFAULT_MISSING_VECTOR_REPORT = PROJECT_ROOT / "data" / "metadata" / "lark_missing_vector_documents.json"


@dataclass(frozen=True)
class CatalogDocument:
    document_key: str
    document_name: str
    document_link: str
    path_titles: list[str]
    oss_object_key: str

    @property
    def lark_path(self) -> str:
        return "/".join([*self.path_titles, self.document_name])

    @property
    def harness_document_dir(self) -> Path:
        """Human-readable Harness asset directory, mirroring the Lark path."""
        path_parts = [sanitize_path_part(part) for part in self.path_titles]
        return HARNESS_KNOWLEDGE_ROOT.joinpath(
            *path_parts,
            sanitize_path_part(self.document_name),
        )

    @property
    def legacy_workspace_dir(self) -> Path:
        """Pre-rebuild hash workspace, removed after this document succeeds."""
        return HARNESS_KNOWLEDGE_ROOT / hashlib.sha1(self.document_key.encode("utf-8")).hexdigest()[:16]

    @property
    def processing_dir(self) -> Path:
        return PROCESSING_ROOT / "lark" / hashlib.sha1(self.document_key.encode("utf-8")).hexdigest()[:16]


def _oss_bucket():
    required = (settings.aliyun_oss_endpoint, settings.aliyun_access_key_id, settings.aliyun_access_key_secret, settings.aliyun_raw_data_bucket)
    if not all(required):
        raise RuntimeError("Missing Aliyun OSS configuration in .env")
    try:
        import oss2
    except ImportError as exc:
        raise RuntimeError(
            "Missing oss2 in this virtual environment. From the project root run: "
            ".venv/bin/python -m pip install 'oss2>=2.19.1,<3.0.0'"
        ) from exc

    return oss2.Bucket(
        oss2.Auth(settings.aliyun_access_key_id, settings.aliyun_access_key_secret),
        settings.aliyun_oss_endpoint,
        settings.aliyun_raw_data_bucket,
    )


def load_catalog_documents(
    limit: int | None,
    document_keys: list[str] | None = None,
) -> list[CatalogDocument]:
    query = """
        SELECT document_key, document_name, document_link, path_titles, oss_object_key
        FROM lark_document_catalog
        WHERE oss_object_key IS NOT NULL AND oss_object_key <> ''
    """
    params: list[Any] = []
    if document_keys:
        query += " AND document_key = ANY(%s)"
        params.append(document_keys)
    query += " ORDER BY path_titles, document_name, document_key"
    if limit is not None:
        query += " LIMIT %s"
    with postgres_connection() as conn, conn.cursor() as cur:
        if limit is not None:
            params.append(limit)
        cur.execute(query, params)
        rows = cur.fetchall()
    return [
        CatalogDocument(
            document_key=str(row["document_key"]), document_name=str(row["document_name"]),
            document_link=str(row["document_link"]), path_titles=list(row["path_titles"] or []),
            oss_object_key=str(row["oss_object_key"]),
        )
        for row in rows
        if Path(str(row["document_name"])).suffix.lower() in SUPPORTED_SUFFIXES
    ]


def download_to_temporary_source(bucket: Any, document: CatalogDocument) -> Path:
    # The source binary is needed only by the parser; do not expose it in the
    # Harness workspace.  It is deleted after TXT/image assets are promoted.
    target = document.processing_dir / "source" / sanitize_path_part(document.document_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    bucket.get_object_to_file(document.oss_object_key, str(target))
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError("OSS download produced an empty file")
    return target


def prepare(document: CatalogDocument, source: Path, *, image_workers: int) -> tuple[Path, Path, list[Any]]:
    items = parse_document(source, image_analysis_workers=image_workers, source="rebuild")
    parsed_dir = processing_document_dir(source)
    source_txt_file = parsed_dir / "txt" / f"{source.stem}.txt"
    if not source_txt_file.is_file():
        raise RuntimeError(f"Parser produced no txt file: {source_txt_file}")
    # Harness receives only model-readable parsing assets.  Replacing this
    # document directory also prevents stale images from an earlier version.
    harness_dir = document.harness_document_dir
    # A previous short-lived layout wrote the source binary at exactly this
    # path.  Remove that file before converting the path into a directory.
    if harness_dir.is_file():
        harness_dir.unlink()
    else:
        shutil.rmtree(harness_dir, ignore_errors=True)
    harness_dir.mkdir(parents=True, exist_ok=True)
    for asset_name in ("txt", "img"):
        asset_dir = parsed_dir / asset_name
        if asset_dir.is_dir():
            shutil.move(str(asset_dir), str(harness_dir / asset_name))
    txt_file = harness_dir / "txt" / source_txt_file.name
    shutil.rmtree(parsed_dir, ignore_errors=True)
    if not txt_file.is_file():
        raise RuntimeError(f"Parser TXT promotion failed: {txt_file}")
    chunks = split_items(items, source_file=source)
    # Stable across document renames and extension changes, so vector upsert
    # removes every prior chunk for this Lark document before inserting new ones.
    file_id = build_file_id("lark", document.document_key)
    for chunk in chunks:
        chunk.metadata.update({
            "file_id": file_id,
            "file_name": document.document_name,
            "file_path": document.lark_path,
            "lark_path": document.lark_path,
            "document_key": document.document_key,
            "document_link": document.document_link,
            "oss_object_key": document.oss_object_key,
        })
    chunk_file = document.processing_dir / "chunk" / f"{source.stem}.chunks.json"
    save_chunks(chunks, chunk_file)
    return txt_file, chunk_file, chunks


def cleanup_temporary_parser_files(document: CatalogDocument) -> None:
    """Remove source binaries and parser work files; preserve chunk artifacts."""
    shutil.rmtree(document.processing_dir / "source", ignore_errors=True)
    shutil.rmtree(document.processing_dir / "parsed", ignore_errors=True)


def load_pending_chunk_files(limit: int | None) -> list[tuple[str, Path, list[Any]]]:
    """Load retained chunk files without downloading or parsing source files."""
    chunk_files = sorted((PROCESSING_ROOT / "lark").glob("*/chunk/*.chunks.json"))
    if limit is not None:
        chunk_files = chunk_files[:limit]
    pending: list[tuple[str, Path, list[Any]]] = []
    for chunk_file in chunk_files:
        chunks = load_chunks(chunk_file)
        if not chunks:
            print(f"[resume] skip empty chunk file {chunk_file}", flush=True)
            continue
        metadata = chunks[0].metadata if chunks else {}
        lark_path = str(metadata.get("lark_path") or metadata.get("file_path") or chunk_file)
        pending.append((lark_path, chunk_file, chunks))
    return pending


def missing_vector_documents(
    documents: list[CatalogDocument], vector_store: VectorStoreService
) -> tuple[list[CatalogDocument], int]:
    """Find catalog documents with no persisted Milvus chunk for their Lark ID."""
    existing_file_ids = vector_store.list_file_ids()
    missing = [
        document
        for document in documents
        if build_file_id("lark", document.document_key) not in existing_file_ids
    ]
    return missing, len(existing_file_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only ingest the first N catalog documents (use 5 for development validation).")
    parser.add_argument("--image-workers", type=int, default=3)
    parser.add_argument("--no-upsert", action="store_true", help="Keep chunk/embedding artifacts and do not write Milvus.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--document-key", action="append", default=[], help="Ingest only this catalog document key; repeat as needed.")
    parser.add_argument("--document-key-file", type=Path, help="UTF-8 file containing one catalog document key per line.")
    parser.add_argument("--use-existing-bm25", action="store_true", help="Use data/processing/global.bm25.json instead of retraining BM25; required for incremental updates.")
    parser.add_argument(
        "--resume-from-chunks",
        action="store_true",
        help="Skip OSS download and parsing; embed/upsert retained data/processing/lark/*/chunk/*.chunks.json files.",
    )
    parser.add_argument(
        "--repair-missing-vectors",
        action="store_true",
        help="Scan every catalog document against Milvus and re-ingest only documents with no persisted chunks.",
    )
    parser.add_argument(
        "--missing-vector-report",
        type=Path,
        default=DEFAULT_MISSING_VECTOR_REPORT,
        help="JSON report written by --repair-missing-vectors.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.resume_from_chunks and (args.document_key or args.document_key_file or args.repair_missing_vectors):
        raise SystemExit("--resume-from-chunks cannot be combined with document filters or --repair-missing-vectors")
    if args.repair_missing_vectors and (args.document_key or args.document_key_file):
        raise SystemExit("--repair-missing-vectors always scans the full catalog and cannot be combined with document filters")

    prepared: list[tuple[str, Path, list[Any]]] = []
    failures: list[tuple[str, str]] = []
    vector_store: VectorStoreService | None = None
    existing_vector_file_id_count = 0
    if args.resume_from_chunks:
        prepared = load_pending_chunk_files(args.limit)
        if not prepared:
            raise SystemExit("No retained chunk files found under data/processing/lark")
        print(f"[start] resume_from_chunks={len(prepared)} limit={args.limit or 'all'}", flush=True)
    else:
        selected_keys = list(args.document_key)
        if args.document_key_file:
            selected_keys.extend(
                line.strip() for line in args.document_key_file.read_text(encoding="utf-8").splitlines() if line.strip()
            )
        documents = load_catalog_documents(
            None if args.repair_missing_vectors else args.limit,
            list(dict.fromkeys(selected_keys)) or None,
        )
        if not documents:
            raise SystemExit("No OSS-backed supported documents found in lark_document_catalog")
        if args.repair_missing_vectors:
            catalog_document_count = len(documents)
            print(f"[repair] scan catalog_documents={catalog_document_count}", flush=True)
            vector_store = VectorStoreService()
            documents, existing_vector_file_id_count = missing_vector_documents(documents, vector_store)
            args.missing_vector_report.parent.mkdir(parents=True, exist_ok=True)
            args.missing_vector_report.write_text(
                json.dumps(
                    {
                        "catalog_documents": catalog_document_count,
                        "persisted_file_ids": existing_vector_file_id_count,
                        "missing": [
                            {"document_key": document.document_key, "lark_path": document.lark_path}
                            for document in documents
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if args.limit is not None:
                documents = documents[:args.limit]
            print(
                f"[repair] persisted_file_ids={existing_vector_file_id_count} "
                f"missing_documents={len(documents)} report={args.missing_vector_report}",
                flush=True,
            )
            if not documents:
                print("[repair] all catalog documents already have Milvus chunks", flush=True)
                return 0
        print(f"[start] catalog_documents={len(documents)} limit={args.limit or 'all'}", flush=True)
        bucket = _oss_bucket()
        for index, document in enumerate(documents, 1):
            print(f"[{index}/{len(documents)}] download+parse {document.lark_path}", flush=True)
            try:
                source = download_to_temporary_source(bucket, document)
                accepted, rule = is_acceptable_knowledge_file(source)
                if not accepted:
                    raise ValueError(f"quality rejected: {rule}")
                _txt, chunk_file, chunks = prepare(document, source, image_workers=args.image_workers)
                # Replace only this document's obsolete hash-named workspace after
                # the OSS download and parser output both completed successfully.
                if document.legacy_workspace_dir.is_dir():
                    shutil.rmtree(document.legacy_workspace_dir)
                prepared.append((document.lark_path, chunk_file, chunks))
                print(f"[{index}/{len(documents)}] chunks={len(chunks)}", flush=True)
            except Exception as exc:
                failures.append((document.lark_path, str(exc)))
                print(f"[{index}/{len(documents)}] failed {document.lark_path}: {exc}", flush=True)
                if not args.continue_on_error:
                    raise
            finally:
                cleanup_temporary_parser_files(document)

    if prepared and not args.no_upsert:
        embedding_service = EmbeddingService()
        all_chunks = [chunk for _path, _chunk_file, chunks in prepared for chunk in chunks]
        bm25_file = default_bm25_model_file()
        if args.repair_missing_vectors and existing_vector_file_id_count and not bm25_file.is_file():
            raise RuntimeError(
                "Cannot repair a partial Milvus collection without data/processing/global.bm25.json. "
                "Restore that file before repairing so sparse-vector vocabulary remains stable."
            )
        use_existing_bm25 = args.use_existing_bm25 or (
            (args.resume_from_chunks or args.repair_missing_vectors) and bm25_file.is_file()
        )
        if use_existing_bm25:
            if not bm25_file.is_file():
                raise RuntimeError("--use-existing-bm25 requires data/processing/global.bm25.json")
            bm25_model = load_bm25_embedding_function(bm25_file)
            print(f"[bm25] loaded={bm25_file}", flush=True)
        else:
            bm25_file, bm25_model = embedding_service.save_global_bm25_model_file(all_chunks, bm25_file)
            print(f"[bm25] trained={bm25_file} chunks={len(all_chunks)}", flush=True)
        vector_store = vector_store or VectorStoreService()
        for index, (lark_path, chunk_file, _chunks) in enumerate(prepared, 1):
            embedding_file = chunk_file.parents[1] / "embedding" / chunk_file.name.replace(".chunks.json", ".embeddings.json")
            try:
                print(f"[{index}/{len(prepared)}] embed {lark_path}", flush=True)
                embedding_service.embed_chunk_file(chunk_file, embedding_file, bm25_model=bm25_model, bm25_model_file=bm25_file)
                result = vector_store.upsert_embedding_file(embedding_file)
                print(f"[{index}/{len(prepared)}] upserted={result['upsert_count']}", flush=True)
                # Do not remove parser output; only remove disposable processing data
                # after Milvus confirms the corresponding write.
                embedding_file.unlink(missing_ok=True)
                chunk_file.unlink(missing_ok=True)
                for directory in (embedding_file.parent, chunk_file.parent, chunk_file.parents[1]):
                    try:
                        directory.rmdir()
                    except OSError:
                        pass
            except Exception as exc:
                failures.append((lark_path, str(exc)))
                print(f"[{index}/{len(prepared)}] embedding-or-upsert failed {lark_path}: {exc}", flush=True)
                if not args.continue_on_error:
                    raise

    if prepared and args.no_upsert:
        print("[summary] --no-upsert: chunk artifacts retained; embedding and Milvus stages skipped", flush=True)
    print(f"[summary] prepared={len(prepared)} failures={len(failures)}", flush=True)
    for path, error in failures:
        print(f"[failure] {path}: {error}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
