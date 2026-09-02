"""Parse OSS-backed Lark knowledge, embed it, and upsert it into Milvus.

Original files and parser outputs live in ``data/harness/knowledge``.  Only
chunk and embedding JSON are temporary ``data/processing`` artifacts; they are
removed after their vectors have been successfully written.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.db.postgres import postgres_connection  # noqa: E402
from app.services.chunking.splitter import build_file_id, save_chunks, split_items  # noqa: E402
from app.services.embedding import EmbeddingService, default_bm25_model_file  # noqa: E402
from app.services.parser.parser import parse_document  # noqa: E402
from app.services.parser.paths import processing_document_dir  # noqa: E402
from app.services.vector_store import VectorStoreService  # noqa: E402

SUPPORTED_SUFFIXES = {".docx", ".xlsx", ".pptx", ".pdf"}
HARNESS_KNOWLEDGE_ROOT = PROJECT_ROOT / "data" / "harness" / "knowledge"
PROCESSING_ROOT = PROJECT_ROOT / "data" / "processing"


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
    def workspace_dir(self) -> Path:
        return HARNESS_KNOWLEDGE_ROOT / hashlib.sha1(self.document_key.encode("utf-8")).hexdigest()[:16]

    @property
    def processing_dir(self) -> Path:
        return PROCESSING_ROOT / "lark" / hashlib.sha1(self.document_key.encode("utf-8")).hexdigest()[:16]


def _oss_bucket():
    required = (settings.aliyun_oss_endpoint, settings.aliyun_access_key_id, settings.aliyun_access_key_secret, settings.aliyun_raw_data_bucket)
    if not all(required):
        raise RuntimeError("Missing Aliyun OSS configuration in .env")
    import oss2

    return oss2.Bucket(
        oss2.Auth(settings.aliyun_access_key_id, settings.aliyun_access_key_secret),
        settings.aliyun_oss_endpoint,
        settings.aliyun_raw_data_bucket,
    )


def load_catalog_documents(limit: int | None) -> list[CatalogDocument]:
    query = """
        SELECT document_key, document_name, document_link, path_titles, oss_object_key
        FROM lark_document_catalog
        WHERE oss_object_key IS NOT NULL AND oss_object_key <> ''
        ORDER BY path_titles, document_name, document_key
    """
    if limit is not None:
        query += " LIMIT %s"
    with postgres_connection() as conn, conn.cursor() as cur:
        cur.execute(query, (limit,) if limit is not None else ())
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


def download_to_workspace(bucket: Any, document: CatalogDocument) -> Path:
    target = document.workspace_dir / "original" / document.document_name
    target.parent.mkdir(parents=True, exist_ok=True)
    bucket.get_object_to_file(document.oss_object_key, str(target))
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError("OSS download produced an empty file")
    return target


def prepare(document: CatalogDocument, source: Path, *, image_workers: int) -> tuple[Path, Path, list[Any]]:
    items = parse_document(source, image_analysis_workers=image_workers, source="rebuild")
    parsed_dir = processing_document_dir(source)
    txt_file = parsed_dir / "txt" / f"{source.stem}.txt"
    if not txt_file.is_file():
        raise RuntimeError(f"Parser produced no txt file: {txt_file}")
    chunks = split_items(items, source_file=source)
    file_id = build_file_id(Path(document.document_name).suffix.lstrip(".") or "file", document.document_key)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only ingest the first N catalog documents (use 5 for development validation).")
    parser.add_argument("--image-workers", type=int, default=3)
    parser.add_argument("--no-upsert", action="store_true", help="Keep chunk/embedding artifacts and do not write Milvus.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    documents = load_catalog_documents(args.limit)
    if not documents:
        raise SystemExit("No OSS-backed supported documents found in lark_document_catalog")
    print(f"[start] catalog_documents={len(documents)} limit={args.limit or 'all'}", flush=True)
    bucket = _oss_bucket()
    prepared: list[tuple[CatalogDocument, Path, list[Any]]] = []
    failures: list[tuple[str, str]] = []
    for index, document in enumerate(documents, 1):
        print(f"[{index}/{len(documents)}] download+parse {document.lark_path}", flush=True)
        try:
            source = download_to_workspace(bucket, document)
            _txt, chunk_file, chunks = prepare(document, source, image_workers=args.image_workers)
            prepared.append((document, chunk_file, chunks))
            print(f"[{index}/{len(documents)}] chunks={len(chunks)}", flush=True)
        except Exception as exc:
            failures.append((document.lark_path, str(exc)))
            print(f"[{index}/{len(documents)}] failed {document.lark_path}: {exc}", flush=True)
            if not args.continue_on_error:
                raise

    if prepared and not args.no_upsert:
        embedding_service = EmbeddingService()
        all_chunks = [chunk for _doc, _path, chunks in prepared for chunk in chunks]
        bm25_file, bm25_model = embedding_service.save_global_bm25_model_file(all_chunks, default_bm25_model_file())
        print(f"[bm25] trained={bm25_file} chunks={len(all_chunks)}", flush=True)
        vector_store = VectorStoreService()
        for index, (document, chunk_file, _chunks) in enumerate(prepared, 1):
            embedding_file = document.processing_dir / "embedding" / chunk_file.name.replace(".chunks.json", ".embeddings.json")
            print(f"[{index}/{len(prepared)}] embed {document.lark_path}", flush=True)
            embedding_service.embed_chunk_file(chunk_file, embedding_file, bm25_model=bm25_model, bm25_model_file=bm25_file)
            result = vector_store.upsert_embedding_file(embedding_file)
            print(f"[{index}/{len(prepared)}] upserted={result['upsert_count']}", flush=True)
            # Do not remove parser output; only remove disposable processing data
            # after Milvus confirms the corresponding write.
            embedding_file.unlink(missing_ok=True)
            chunk_file.unlink(missing_ok=True)
            for directory in (embedding_file.parent, chunk_file.parent, document.processing_dir):
                try:
                    directory.rmdir()
                except OSError:
                    pass

    if prepared and args.no_upsert:
        print("[summary] --no-upsert: chunk artifacts retained; embedding and Milvus stages skipped", flush=True)
    print(f"[summary] prepared={len(prepared)} failures={len(failures)}", flush=True)
    for path, error in failures:
        print(f"[failure] {path}: {error}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
