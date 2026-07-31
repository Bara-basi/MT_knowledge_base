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
from app.services.parser.parser import (
    is_generated_standard_pdf_section,
    parse_document,
)
from app.services.parser.paths import processing_document_dir
from app.services.parser.unified_pdf_parser import (
    _is_asme_parent_pdf,
    resolve_pdf_document_sources,
)
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
    force_upsert: bool = False


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
    if _source_suffix(file_path) == ".pdf":
        resolved_already_parsed = False
        resolved_sources = resolve_pdf_document_sources(
            file_path,
            split_if_missing=False,
        )
        if (
            len(resolved_sources) == 1
            and str(resolved_sources[0]) == str(file_path)
            and _is_asme_parent_pdf(file_path)
        ):
            if not parse:
                raise FileNotFoundError(
                    f"ASME split edition is required when parsing is disabled: {file_path}"
                )
            # parse_document performs split-only preprocessing when needed and
            # parses every resulting small PDF through the unified pipeline.
            parse_document(
                file_path,
                image_analysis_workers=image_analysis_workers,
            )
            resolved_already_parsed = True
            resolved_sources = resolve_pdf_document_sources(
                file_path,
                split_if_missing=False,
            )
        if len(resolved_sources) != 1 or str(resolved_sources[0]) != str(file_path):
            return [
                prepare_document(
                    resolved_source,
                    image_analysis_workers=image_analysis_workers,
                    rebuild=rebuild,
                    parse=False if resolved_already_parsed else parse,
                )
                for resolved_source in resolved_sources
            ]
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
    """Backward-compatible name routed to the unified PDF preparation path."""

    return prepare_documents(
        file_path,
        image_analysis_workers=image_analysis_workers,
        rebuild=rebuild,
        parse=parse,
    )


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
        output_path = PurePosixPath(str(section.get("output_path") or "").replace("\\", "/"))
        if output_path.stem == txt_file.stem:
            source_uri = str(section.get("source_uri") or "").strip()
            return parse_raw_document_reference(source_uri).uri if source_uri else None
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
        if skip_existing_upsert and not prepared.force_upsert and embedding_file_exists_in_vector_store(
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
    generated_section_prefixes = {
        str(PurePosixPath(reference.object_name).parent).rstrip("/")
        for reference in objects
        if is_generated_standard_pdf_section(reference.object_name)
        and is_supported_document_name(reference.object_name)
    }
    return sorted(
        reference.uri
        for reference in objects
        if is_supported_document_name(reference.object_name)
        and (
            is_generated_standard_pdf_section(reference.object_name)
            or not _asme_parent_has_split_edition(
                reference.object_name,
                generated_section_prefixes,
            )
        )
    )


def _asme_parent_has_split_edition(
    object_name: str,
    generated_section_prefixes: set[str],
) -> bool:
    path = PurePosixPath(object_name)
    if not _is_asme_parent_pdf(path.as_posix()):
        return False
    expected_prefix = (path.parent / f"{path.stem}(切分版)").as_posix()
    return expected_prefix in generated_section_prefixes


def is_supported_document_name(name: str) -> bool:
    path = PurePosixPath(name)
    return path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith("~$")


def is_standard_pdf_document(path: str | Path) -> bool:
    return _is_asme_parent_pdf(path)


def document_file_id(file_path: str | Path) -> str:
    suffix = _source_suffix(file_path)
    file_type = "doc" if suffix == ".docx" else suffix.lstrip(".") or "unknown"
    return build_file_id(file_type, _source_stem(file_path) or "unknown")


def _chunks_reference_source(chunks: list, source_file: str | Path) -> bool:
    if not chunks:
        return True
    expected = str(source_file).replace("\\", "/").strip()
    return all(
        str(getattr(chunk, "metadata", {}).get("file_path") or "")
        .replace("\\", "/")
        .strip()
        == expected
        for chunk in chunks
    )


def _embedding_file_references_source(
    embedding_file: Path,
    source_file: str | Path,
) -> bool:
    records = load_embedding_records(embedding_file)
    if not records:
        return True
    expected = str(source_file).replace("\\", "/").strip()
    return all(
        str((record.get("metadata") or {}).get("file_path") or "")
        .replace("\\", "/")
        .strip()
        == expected
        for record in records
    )


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
