from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.chunking.splitter import save_chunks, split_items
from app.services.embedding import (
    EmbeddingService,
    default_bm25_model_file,
    load_bm25_embedding_function,
)
from app.services.parser.parser import parse_document
from app.services.vector_store import VectorStoreService


SUPPORTED_EXTENSIONS = {".docx", ".pptx", ".xlsx"}
BM25_MODES = {"auto", "existing", "train-input"}


@dataclass
class IngestionResult:
    file_path: Path
    txt_file: Path
    chunk_file: Path
    embedding_file: Path
    chunk_count: int
    upsert_count: int


@dataclass
class PreparedDocument:
    file_path: Path
    txt_file: Path
    chunk_file: Path
    embedding_file: Path
    chunks: list


def main() -> None:
    _configure_utf8_stdio()

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
    parser.add_argument(
        "--bm25-model-file",
        default=str(default_bm25_model_file()),
        help="BM25 model JSON used for sparse vectors. Default: data/processing/global.bm25.json.",
    )
    parser.add_argument(
        "--bm25-mode",
        choices=sorted(BM25_MODES),
        default="auto",
        help=(
            "BM25 handling: auto loads an existing model or trains one if missing; "
            "existing requires an existing model; train-input retrains from this input batch. "
            "Default: auto."
        ),
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
    print(f"BM25 mode: {args.bm25_mode}")

    prepared_documents: list[PreparedDocument] = []
    failures: list[tuple[Path, Exception]] = []

    for index, file_path in enumerate(document_files, start=1):
        print(f"\n[{index}/{len(document_files)}] Parsing {file_path}")
        try:
            prepared = prepare_document(
                file_path,
                image_analysis_workers=args.image_workers,
            )
        except Exception as exc:
            failures.append((file_path, exc))
            print(f"Failed: {exc}")
            if not args.continue_on_error:
                raise
            continue

        prepared_documents.append(prepared)
        print(f"Text output: {prepared.txt_file}")
        print(f"Chunk output: {prepared.chunk_file}")
        print(f"Chunks: {len(prepared.chunks)}")

    results: list[IngestionResult] = []
    if prepared_documents:
        bm25_model_file, bm25_model, bm25_action = resolve_bm25_model(
            prepared_documents,
            embedding_service=embedding_service,
            bm25_model_file=Path(args.bm25_model_file),
            mode=args.bm25_mode,
        )
        print(f"\nBM25 model: {bm25_model_file}")
        print(f"BM25 action: {bm25_action}")

        for index, prepared in enumerate(prepared_documents, start=1):
            print(f"\n[{index}/{len(prepared_documents)}] Embedding {prepared.file_path}")
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
                print(f"Failed: {exc}")
                if not args.continue_on_error:
                    raise
                continue

            results.append(result)
            print(f"Embedding output: {result.embedding_file}")
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


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def ingest_document(
    file_path: Path,
    *,
    embedding_service: EmbeddingService,
    vector_store_service: VectorStoreService | None,
    flush: bool,
    image_analysis_workers: int,
    bm25_model=None,
    bm25_model_file: Path | None = None,
) -> IngestionResult:
    prepared = prepare_document(
        file_path,
        image_analysis_workers=image_analysis_workers,
    )
    bm25_model_file = bm25_model_file or default_bm25_model_file()
    if bm25_model is None:
        if Path(bm25_model_file).exists():
            bm25_model = load_bm25_embedding_function(bm25_model_file)
        else:
            bm25_model_file, bm25_model = embedding_service.save_global_bm25_model_file(
                prepared.chunks,
                bm25_model_file,
            )
    return embed_prepared_document(
        prepared,
        embedding_service=embedding_service,
        vector_store_service=vector_store_service,
        flush=flush,
        bm25_model=bm25_model,
        bm25_model_file=bm25_model_file,
    )


def prepare_document(
    file_path: Path,
    *,
    image_analysis_workers: int,
) -> PreparedDocument:
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

    return PreparedDocument(
        file_path=file_path,
        txt_file=txt_file,
        chunk_file=chunk_file,
        embedding_file=embedding_file,
        chunks=chunks,
    )


def resolve_bm25_model(
    prepared_documents: list[PreparedDocument],
    *,
    embedding_service: EmbeddingService,
    bm25_model_file: Path,
    mode: str,
):
    if mode not in BM25_MODES:
        raise ValueError(f"Unsupported BM25 mode: {mode}. Choose from: {sorted(BM25_MODES)}")

    if mode in {"auto", "existing"} and bm25_model_file.exists():
        bm25_model = load_bm25_embedding_function(bm25_model_file)
        return bm25_model_file, bm25_model, "loaded existing model"

    if mode == "existing":
        raise FileNotFoundError(f"BM25 model file not found: {bm25_model_file}")

    all_chunks = [
        chunk
        for prepared in prepared_documents
        for chunk in prepared.chunks
    ]
    bm25_model_file, bm25_model = embedding_service.save_global_bm25_model_file(
        all_chunks,
        bm25_model_file,
    )
    return bm25_model_file, bm25_model, "trained from input batch"


def embed_prepared_document(
    prepared: PreparedDocument,
    *,
    embedding_service: EmbeddingService,
    vector_store_service: VectorStoreService | None,
    flush: bool,
    bm25_model,
    bm25_model_file: Path | None,
) -> IngestionResult:
    embedding_service.embed_chunk_file(
        prepared.chunk_file,
        prepared.embedding_file,
        bm25_model=bm25_model,
        bm25_model_file=bm25_model_file,
    )

    upsert_count = 0
    if vector_store_service is not None:
        upsert_result = vector_store_service.upsert_embedding_file(
            prepared.embedding_file,
            flush=flush,
        )
        upsert_count = int(upsert_result["upsert_count"])

    return IngestionResult(
        file_path=prepared.file_path,
        txt_file=prepared.txt_file,
        chunk_file=prepared.chunk_file,
        embedding_file=prepared.embedding_file,
        chunk_count=len(prepared.chunks),
        upsert_count=upsert_count,
    )


if __name__ == "__main__":
    main()
