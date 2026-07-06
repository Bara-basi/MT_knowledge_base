from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.chunking.splitter import build_file_id, save_chunks, split_items
from app.services.chunking.splitter import load_items_from_txt
from app.db.minio import list_raw_document_objects, parse_raw_document_reference
from app.services.embedding import (
    EmbeddingService,
    default_bm25_model_file,
    load_chunks,
    load_bm25_embedding_function,
)
from app.services.parser.parser import parse_document
from app.services.parser.paths import processing_document_dir
from app.services.vector_store import VectorStoreService, load_embedding_records


SUPPORTED_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".pdf"}
BM25_MODES = {"auto", "existing", "train-input"}


@dataclass
class IngestionResult:
    file_path: str | Path
    txt_file: Path
    chunk_file: Path
    embedding_file: Path
    chunk_count: int
    upsert_count: int
    upsert_skipped: bool = False


@dataclass
class PreparedDocument:
    file_path: str | Path
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
        nargs="?",
        default="",
        help="MinIO document object or prefix to ingest. Default: bucket root.",
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
        "--continue",
        dest="continue_existing",
        action="store_true",
        help=(
            "Resume from existing artifacts. Existing txt, chunk, and embedding files "
            "are reused; missing later stages are generated and upsert can still run."
        ),
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

    input_path = args.input_path
    document_files = find_document_files(input_path, recursive=args.recursive)
    if not document_files:
        raise SystemExit(f"No supported documents found under: {input_path}")

    embedding_service = EmbeddingService()
    vector_store_service = None if args.no_upsert else VectorStoreService()

    print(f"Input: {input_path}")
    print(f"Documents: {len(document_files)}")
    print(f"Upsert: {'disabled' if args.no_upsert else 'enabled'}")
    print(f"BM25 mode: {args.bm25_mode}")
    print(f"Continue: {'enabled' if args.continue_existing else 'disabled'}")

    prepared_documents: list[PreparedDocument] = []
    failures: list[tuple[str | Path, Exception]] = []

    for index, file_path in enumerate(document_files, start=1):
        print(f"\n[{index}/{len(document_files)}] Parsing {file_path}")
        try:
            prepared_batch = prepare_documents(
                file_path,
                image_analysis_workers=args.image_workers,
                rebuild=not args.continue_existing,
            )
        except Exception as exc:
            failures.append((file_path, exc))
            print(f"Failed: {exc}")
            if not args.continue_on_error:
                raise
            continue

        prepared_documents.extend(prepared_batch)
        print(f"Prepared outputs: {len(prepared_batch)}")
        print(f"Chunks: {sum(len(prepared.chunks) for prepared in prepared_batch)}")
        if prepared_batch:
            print(f"First text output: {prepared_batch[0].txt_file}")
            print(f"First chunk output: {prepared_batch[0].chunk_file}")

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
                    rebuild=not args.continue_existing,
                    skip_existing_upsert=args.continue_existing,
                )
            except Exception as exc:
                failures.append((prepared.file_path, exc))
                print(f"Failed: {exc}")
                if not args.continue_on_error:
                    raise
                continue

            results.append(result)
            print(f"Embedding output: {result.embedding_file}")
            if result.upsert_skipped:
                print("Upsert skipped: existing file_id found in Milvus")
            else:
                print(f"Upserted: {result.upsert_count}")

    print("\nSummary")
    print(f"Succeeded documents: {len(results)}")
    print(f"Failed documents: {len(failures)}")
    print(f"Total chunks: {sum(result.chunk_count for result in results)}")
    print(f"Total upserted: {sum(result.upsert_count for result in results)}")
    print(f"Skipped upserts: {sum(1 for result in results if result.upsert_skipped)}")

    if failures:
        print("\nFailures")
        for file_path, exc in failures:
            print(f"- {file_path}: {exc}")
        raise SystemExit(1)


def find_document_files(input_path: str | Path, *, recursive: bool) -> list[str]:
    objects = list_raw_document_objects(str(input_path or ""), recursive=recursive)
    skipped_prefixes = split_version_source_prefixes(
        reference.object_name for reference in objects
    )
    return sorted(
        reference.uri
        for reference in objects
        if is_supported_document_name(reference.object_name)
        and not is_shadowed_by_split_version(reference.object_name, skipped_prefixes)
    )


def is_supported_document(path: str | Path) -> bool:
    return is_supported_document_name(_source_name(path))


def is_supported_document_name(name: str) -> bool:
    path = PurePosixPath(name)
    return path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith("~$")


def is_standard_pdf_document(path: str | Path) -> bool:
    normalized = str(path).replace("\\", "/")
    return (
        _source_suffix(path) == ".pdf"
        and "标准文档" in normalized
        and "(切分版)" not in normalized
    )


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def ingest_document(
    file_path: str | Path,
    *,
    embedding_service: EmbeddingService,
    vector_store_service: VectorStoreService | None,
    flush: bool,
    image_analysis_workers: int,
    bm25_model=None,
    bm25_model_file: Path | None = None,
    rebuild: bool = True,
    skip_existing_upsert: bool = False,
) -> IngestionResult:
    prepared = prepare_document(
        file_path,
        image_analysis_workers=image_analysis_workers,
        rebuild=rebuild,
    )
    bm25_model_file = bm25_model_file or default_bm25_model_file()
    if prepared.chunks and bm25_model is None:
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
        rebuild=rebuild,
        skip_existing_upsert=skip_existing_upsert,
    )


def prepare_documents(
    file_path: str | Path,
    *,
    image_analysis_workers: int,
    rebuild: bool = True,
    parse: bool = True,
) -> list[PreparedDocument]:
    if is_standard_pdf_document(file_path):
        return prepare_standard_pdf_sections(
            file_path,
            image_analysis_workers=image_analysis_workers,
            rebuild=rebuild,
            parse=parse,
        )
    return [
        prepare_document(
            file_path,
            image_analysis_workers=image_analysis_workers,
            rebuild=rebuild,
            parse=parse,
        )
    ]


def prepare_document(
    file_path: str | Path,
    *,
    image_analysis_workers: int,
    rebuild: bool = True,
    parse: bool = True,
) -> PreparedDocument:
    document_name = _source_stem(file_path)
    processing_dir = processing_document_dir(file_path)

    txt_file = processing_dir / "txt" / f"{document_name}.txt"
    chunk_file = processing_dir / "chunk" / f"{document_name}.chunks.json"
    embedding_file = processing_dir / "embedding" / f"{document_name}.embeddings.json"

    if not rebuild and chunk_file.exists():
        chunks = load_chunks(chunk_file)
    else:
        if not parse:
            if not txt_file.exists():
                raise FileNotFoundError(
                    f"Parsed txt file is required when parse is disabled: {txt_file}"
                )
            parsed_items = load_items_from_txt(txt_file)
        elif not rebuild and txt_file.exists():
            parsed_items = load_items_from_txt(txt_file)
        else:
            parsed_items = parse_document(
                file_path,
                image_analysis_workers=image_analysis_workers,
            )

        chunks = split_items(parsed_items, source_file=file_path)
        save_chunks(chunks, chunk_file)

    return PreparedDocument(
        file_path=file_path,
        txt_file=txt_file,
        chunk_file=chunk_file,
        embedding_file=embedding_file,
        chunks=chunks,
    )


def prepare_standard_pdf_sections(
    file_path: str | Path,
    *,
    image_analysis_workers: int,
    rebuild: bool = True,
    parse: bool = True,
) -> list[PreparedDocument]:
    processing_dir = processing_document_dir(file_path)

    if parse and rebuild:
        parse_document(
            file_path,
            image_analysis_workers=image_analysis_workers,
        )

    section_txt_files = find_standard_section_txt_files(processing_dir)
    if parse and not rebuild and not section_txt_files:
        parse_document(
            file_path,
            image_analysis_workers=image_analysis_workers,
        )
        section_txt_files = find_standard_section_txt_files(processing_dir)
    if not section_txt_files:
        raise FileNotFoundError(f"No standard section txt files found under: {processing_dir}")

    prepared_documents: list[PreparedDocument] = []
    for txt_file in section_txt_files:
        section_dir = txt_file.parent.parent
        section_pdf = section_pdf_for_txt(txt_file)
        source_file = standard_section_source_for_txt(txt_file) or (
            section_pdf if section_pdf.exists() else txt_file
        )
        chunk_file = section_dir / "chunk" / f"{txt_file.stem}.chunks.json"
        embedding_file = section_dir / "embedding" / f"{txt_file.stem}.embeddings.json"

        if not rebuild and chunk_file.exists():
            chunks = load_chunks(chunk_file)
        else:
            parsed_items = load_items_from_txt(txt_file)
            chunks = split_items(parsed_items, source_file=source_file)
            save_chunks(chunks, chunk_file)

        prepared_documents.append(
            PreparedDocument(
                file_path=source_file,
                txt_file=txt_file,
                chunk_file=chunk_file,
                embedding_file=embedding_file,
                chunks=chunks,
            )
        )

    return prepared_documents


def find_standard_section_txt_files(processing_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in processing_dir.glob("*/txt/*.txt")
        if path.is_file()
    )


def section_pdf_for_txt(txt_file: Path) -> Path:
    section_dir = txt_file.parent.parent
    return section_dir / "pdf" / f"{txt_file.stem}.pdf"


def standard_section_source_for_txt(txt_file: Path) -> str | None:
    manifest_path = _standard_manifest_path_for_txt(txt_file)
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for section in payload.get("sections", []):
        if not isinstance(section, dict):
            continue
        output_path = Path(str(section.get("output_path") or ""))
        if output_path.stem == txt_file.stem:
            source_uri = str(section.get("source_uri") or "").strip()
            return source_uri or None
    return None


def _standard_manifest_path_for_txt(txt_file: Path) -> Path:
    section_dir = txt_file.parent.parent
    return section_dir.parent / "manifest.json"


def split_version_source_prefixes(object_names) -> set[str]:
    prefixes: set[str] = set()
    for object_name in object_names:
        parts = PurePosixPath(str(object_name).replace("\\", "/")).parts
        for index, part in enumerate(parts[:-1]):
            if part.endswith("(切分版)"):
                original = part[: -len("(切分版)")]
                prefixes.add("/".join((*parts[:index], original)))
    return {prefix for prefix in prefixes if prefix}


def is_shadowed_by_split_version(object_name: str, skipped_prefixes: set[str]) -> bool:
    normalized = str(object_name).replace("\\", "/").strip("/")
    if "(切分版)" in normalized:
        return False
    for prefix in skipped_prefixes:
        if normalized.startswith(f"{prefix}/"):
            return True
        original_file = f"{prefix}{PurePosixPath(normalized).suffix}"
        if normalized == original_file:
            return True
    return False


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
    if not all_chunks:
        return bm25_model_file, None, "skipped: no chunks in input batch"

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
    rebuild: bool = True,
    skip_existing_upsert: bool = False,
) -> IngestionResult:
    if rebuild or not prepared.embedding_file.exists():
        embedding_service.embed_chunk_file(
            prepared.chunk_file,
            prepared.embedding_file,
            bm25_model=bm25_model,
            bm25_model_file=bm25_model_file,
        )

    upsert_count = 0
    upsert_skipped = False
    if vector_store_service is not None:
        if skip_existing_upsert and embedding_file_exists_in_vector_store(
            prepared.embedding_file,
            vector_store_service,
        ):
            upsert_skipped = True
        else:
            upsert_result = vector_store_service.upsert_embedding_file(
                prepared.embedding_file,
                flush=flush,
                delete_file_ids=[] if prepared.chunks else [document_file_id(prepared.file_path)],
            )
            upsert_count = int(upsert_result["upsert_count"])

    return IngestionResult(
        file_path=prepared.file_path,
        txt_file=prepared.txt_file,
        chunk_file=prepared.chunk_file,
        embedding_file=prepared.embedding_file,
        chunk_count=len(prepared.chunks),
        upsert_count=upsert_count,
        upsert_skipped=upsert_skipped,
    )


def embedding_file_exists_in_vector_store(
    embedding_file: Path,
    vector_store_service: VectorStoreService,
) -> bool:
    file_ids = embedding_file_file_ids(embedding_file)
    return bool(file_ids) and all(vector_store_service.has_file_id(file_id) for file_id in file_ids)


def embedding_file_file_ids(embedding_file: Path) -> list[str]:
    records = load_embedding_records(embedding_file)
    file_ids: list[str] = []
    seen: set[str] = set()
    for record in records:
        metadata = dict(record.get("metadata") or {})
        file_id = str(record.get("file_id") or metadata.get("file_id") or "").strip()
        if not file_id or file_id in seen:
            continue
        seen.add(file_id)
        file_ids.append(file_id)
    return file_ids


def document_file_id(file_path: str | Path) -> str:
    suffix = _source_suffix(file_path)
    file_type = "doc" if suffix == ".docx" else suffix.lstrip(".") or "unknown"
    return build_file_id(file_type, _source_stem(file_path) or "unknown")


def _source_parts(path: str | Path) -> PurePosixPath:
    value = str(path)
    normalized = value.replace("\\", "/")
    parsed = urlparse(normalized)
    if parsed.scheme in {"minio", "s3"} or "data/raw/" in normalized.lower():
        reference = parse_raw_document_reference(value)
        return PurePosixPath(reference.object_name)
    return PurePosixPath(Path(value).as_posix())


def _source_name(path: str | Path) -> str:
    return _source_parts(path).name


def _source_stem(path: str | Path) -> str:
    return _source_parts(path).stem


def _source_suffix(path: str | Path) -> str:
    return _source_parts(path).suffix.lower()


if __name__ == "__main__":
    main()
