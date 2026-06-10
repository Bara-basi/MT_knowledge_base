from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.milvus import (
    MilvusCollectionConfig,
    drop_chunk_collection,
    ensure_chunk_collection,
    get_milvus_client,
)
from scripts.ingestion.ingest_documents import (
    IngestionResult,
    PreparedDocument,
    embed_prepared_document,
    find_document_files,
    prepare_document,
)
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService


@dataclass
class ReindexSummary:
    input_path: Path
    collection_name: str
    dropped: bool
    document_count: int
    succeeded_count: int
    failed_count: int
    total_chunks: int
    total_upserted: int


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_utf8_stdio()

    parser = argparse.ArgumentParser(
        description=(
            "Drop the Milvus chunk collection, recreate it with the current schema, "
            "and reingest all supported documents."
        ),
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="data/raw",
        help="Document file or folder to reindex. Default: data/raw.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan folders recursively. Enabled by default.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue reindexing other documents when one document fails.",
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=3,
        help="Max concurrent image description API calls per document. Default: 3.",
    )
    parser.add_argument(
        "--no-flush",
        action="store_true",
        help="Skip Milvus flush after each document upsert. Useful for larger batches.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    document_files = find_document_files(input_path, recursive=args.recursive)
    if not document_files:
        raise SystemExit(f"No supported documents found under: {input_path}")

    config = MilvusCollectionConfig()
    client = get_milvus_client()

    print(f"Input: {input_path}")
    print(f"Documents: {len(document_files)}")
    print(f"Collection: {config.name}")
    print("Dropping existing collection...")
    drop_result = drop_chunk_collection(client=client, config=config)
    print(f"Dropped: {drop_result['dropped']}")

    print("Creating collection with current schema...")
    ensure_result = ensure_chunk_collection(client=client, config=config)
    print(f"Created: {ensure_result['created']}")

    embedding_service = EmbeddingService()
    vector_store_service = VectorStoreService(client=client, config=config)

    prepared_documents: list[PreparedDocument] = []
    failures: list[tuple[Path, Exception]] = []

    for index, file_path in enumerate(document_files, start=1):
        print(f"\n[{index}/{len(document_files)}] Parsing {file_path}", flush=True)
        try:
            prepared = prepare_document(
                file_path,
                image_analysis_workers=args.image_workers,
            )
        except Exception as exc:
            failures.append((file_path, exc))
            print(f"Failed: {type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_error:
                raise
            continue

        prepared_documents.append(prepared)
        print(f"Text output: {prepared.txt_file}")
        print(f"Chunk output: {prepared.chunk_file}")
        print(f"Chunks: {len(prepared.chunks)}")

    results: list[IngestionResult] = []
    if prepared_documents:
        all_chunks = [
            chunk
            for prepared in prepared_documents
            for chunk in prepared.chunks
        ]
        print("\nTraining global BM25 model...")
        bm25_model_file, bm25_model = embedding_service.save_global_bm25_model_file(all_chunks)
        print(f"Global BM25 model: {bm25_model_file}")

        for index, prepared in enumerate(prepared_documents, start=1):
            print(f"\n[{index}/{len(prepared_documents)}] Embedding {prepared.file_path}", flush=True)
            try:
                result = embed_prepared_document(
                    prepared,
                    embedding_service=embedding_service,
                    vector_store_service=vector_store_service,
                    flush=not args.no_flush,
                    bm25_model=bm25_model,
                    bm25_model_file=bm25_model_file,
                )
            except Exception as exc:
                failures.append((prepared.file_path, exc))
                print(f"Failed: {type(exc).__name__}: {exc}", flush=True)
                if not args.continue_on_error:
                    raise
                continue

            results.append(result)
            print(f"Embedding output: {result.embedding_file}")
            print(f"Upserted: {result.upsert_count}")

    if args.no_flush:
        print("\nFlushing collection...")
        client.flush(collection_name=config.name)

    summary = ReindexSummary(
        input_path=input_path,
        collection_name=config.name,
        dropped=bool(drop_result["dropped"]),
        document_count=len(document_files),
        succeeded_count=len(results),
        failed_count=len(failures),
        total_chunks=sum(result.chunk_count for result in results),
        total_upserted=sum(result.upsert_count for result in results),
    )

    print("\nSummary")
    print(f"Input: {summary.input_path}")
    print(f"Collection: {summary.collection_name}")
    print(f"Dropped existing collection: {summary.dropped}")
    print(f"Documents: {summary.document_count}")
    print(f"Succeeded documents: {summary.succeeded_count}")
    print(f"Failed documents: {summary.failed_count}")
    print(f"Total chunks: {summary.total_chunks}")
    print(f"Total upserted: {summary.total_upserted}")

    if failures:
        print("\nFailures")
        for file_path, exc in failures:
            print(f"- {file_path}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
