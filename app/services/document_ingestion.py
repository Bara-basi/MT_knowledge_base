from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from app.db.minio import list_raw_document_objects, parse_raw_document_reference
from app.services.chunking.splitter import (
    build_file_id,
    load_items_from_txt,
    save_chunks,
    split_items,
)
from app.services.embedding import (
    EmbeddingService,
    default_bm25_model_file,
    load_bm25_embedding_function,
    load_chunks,
)
from app.services.parser.parser import parse_document
from app.services.parser.paths import processing_document_dir
from app.services.vector_store import VectorStoreService, load_embedding_records


SUPPORTED_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".pdf"}


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
                raise FileNotFoundError(f"Parsed txt file is required when parse is disabled: {txt_file}")
            parsed_items = load_items_from_txt(txt_file)
        elif not rebuild and txt_file.exists():
            parsed_items = load_items_from_txt(txt_file)
        else:
            parsed_items = parse_document(file_path, image_analysis_workers=image_analysis_workers)
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
        parse_document(file_path, image_analysis_workers=image_analysis_workers)

    section_txt_files = find_standard_section_txt_files(processing_dir)
    if parse and not rebuild and not section_txt_files:
        parse_document(file_path, image_analysis_workers=image_analysis_workers)
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
    return sorted(path for path in processing_dir.glob("*/txt/*.txt") if path.is_file())


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


def find_document_files(input_path: str | Path, *, recursive: bool) -> list[str]:
    objects = list_raw_document_objects(str(input_path or ""), recursive=recursive)
    skipped_prefixes = split_version_source_prefixes(reference.object_name for reference in objects)
    return sorted(
        reference.uri
        for reference in objects
        if is_supported_document_name(reference.object_name)
        and not is_shadowed_by_split_version(reference.object_name, skipped_prefixes)
    )


def is_supported_document_name(name: str) -> bool:
    path = PurePosixPath(name)
    return path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith("~$")


def is_standard_pdf_document(path: str | Path) -> bool:
    normalized = str(path).replace("\\", "/")
    return (
        _source_suffix(path) == ".pdf"
        and "鏍囧噯鏂囨。" in normalized
        and "(鍒囧垎鐗?" not in normalized
    )


def split_version_source_prefixes(object_names) -> set[str]:
    prefixes: set[str] = set()
    for object_name in object_names:
        parts = PurePosixPath(str(object_name).replace("\\", "/")).parts
        for index, part in enumerate(parts[:-1]):
            if part.endswith("(鍒囧垎鐗?"):
                original = part[: -len("(鍒囧垎鐗?")]
                prefixes.add("/".join((*parts[:index], original)))
    return {prefix for prefix in prefixes if prefix}


def is_shadowed_by_split_version(object_name: str, skipped_prefixes: set[str]) -> bool:
    normalized = str(object_name).replace("\\", "/").strip("/")
    if "(鍒囧垎鐗?" in normalized:
        return False
    for prefix in skipped_prefixes:
        if normalized.startswith(f"{prefix}/"):
            return True
        original_file = f"{prefix}{PurePosixPath(normalized).suffix}"
        if normalized == original_file:
            return True
    return False


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


def _source_stem(path: str | Path) -> str:
    return _source_parts(path).stem


def _source_suffix(path: str | Path) -> str:
    return _source_parts(path).suffix.lower()
