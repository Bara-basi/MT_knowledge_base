from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.chunking.splitter import save_chunks, split_items
from app.services.embedding import EmbeddingService
from app.services.parser.parser import parse_document
from app.services.vector_store import VectorStoreService


SUPPORTED_EXTENSIONS = {".docx"}


@dataclass
class IngestionResult:
    file_path: Path
    txt_file: Path
    chunk_file: Path
    embedding_file: Path
    chunk_count: int
    upsert_count: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch parse, chunk, embed, and upsert documents into Milvus.",
    )
    parser.add_argument(
        "input_path",
        help="Document file or folder to ingest, for example data/raw/process_guide.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan folders recursively. Enabled by default.",
    )
    parser.add_argument(
        "--no-upsert",
        action="store_true",
        help="Generate processing artifacts but do not write vectors into Milvus.",
    )
    parser.add_argument(
        "--no-flush",
        action="store_true",
        help="Skip Milvus flush after each upsert. Useful for larger batches.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue ingesting other documents when one document fails.",
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=3,
        help="Max concurrent image description API calls per document. Default: 3.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    document_files = find_document_files(input_path, recursive=args.recursive)
    if not document_files:
        raise SystemExit(f"No supported documents found under: {input_path}")

    embedding_service = EmbeddingService()
    vector_store_service = None if args.no_upsert else VectorStoreService()

    print(f"Input: {input_path}")
    print(f"Documents: {len(document_files)}")
    print(f"Upsert: {'disabled' if args.no_upsert else 'enabled'}")

    results: list[IngestionResult] = []
    failures: list[tuple[Path, Exception]] = []

    for index, file_path in enumerate(document_files, start=1):
        print(f"\n[{index}/{len(document_files)}] {file_path}")
        try:
            result = ingest_document(
                file_path,
                embedding_service=embedding_service,
                vector_store_service=vector_store_service,
                flush=not args.no_flush,
                image_analysis_workers=args.image_workers,
            )
        except Exception as exc:
            failures.append((file_path, exc))
            print(f"Failed: {exc}")
            if not args.continue_on_error:
                raise
            continue

        results.append(result)
        print(f"Text output: {result.txt_file}")
        print(f"Chunk output: {result.chunk_file}")
        print(f"Embedding output: {result.embedding_file}")
        print(f"Chunks: {result.chunk_count}")
        print(f"Upserted: {result.upsert_count}")

    print("\nSummary")
    print(f"Succeeded documents: {len(results)}")
    print(f"Failed documents: {len(failures)}")
    print(f"Total chunks: {sum(result.chunk_count for result in results)}")
    print(f"Total upserted: {sum(result.upsert_count for result in results)}")

    if failures:
        print("\nFailures")
        for file_path, exc in failures:
            print(f"- {file_path}: {exc}")
        raise SystemExit(1)


def find_document_files(input_path: Path, *, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if is_supported_document(input_path) else []

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Input path must be a file or folder: {input_path}")

    iterator: Iterable[Path]
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and is_supported_document(path)
    )


def is_supported_document(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith("~$")


def ingest_document(
    file_path: Path,
    *,
    embedding_service: EmbeddingService,
    vector_store_service: VectorStoreService | None,
    flush: bool,
    image_analysis_workers: int,
) -> IngestionResult:
    parsed_items = parse_document(
        file_path,
        image_analysis_workers=image_analysis_workers,
    )
    document_name = file_path.stem
    processing_dir = Path("data") / "processing" / document_name

    txt_file = processing_dir / "txt" / f"{document_name}.txt"
    chunk_file = processing_dir / "chunk" / f"{document_name}.chunks.json"
    embedding_file = processing_dir / "embedding" / f"{document_name}.embeddings.json"

    chunks = split_items(parsed_items, source_file=file_path)
    save_chunks(chunks, chunk_file)
    embedding_service.embed_chunk_file(chunk_file, embedding_file)

    upsert_count = 0
    if vector_store_service is not None:
        upsert_result = vector_store_service.upsert_embedding_file(
            embedding_file,
            flush=flush,
        )
        upsert_count = int(upsert_result["upsert_count"])

    return IngestionResult(
        file_path=file_path,
        txt_file=txt_file,
        chunk_file=chunk_file,
        embedding_file=embedding_file,
        chunk_count=len(chunks),
        upsert_count=upsert_count,
    )


if __name__ == "__main__":
    main()
