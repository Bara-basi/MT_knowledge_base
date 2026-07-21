from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from app.db.minio import download_raw_document_to_file, parse_raw_document_reference
from app.services.parser.paths import processing_document_dir
from app.services.processed_document_assets import synchronize_processed_assets


def parse_document(
    file_path: str | Path,
    *,
    image_analysis_workers: int = 3,
):
    """Route an input document to the matching parser by file extension."""
    source = str(file_path)
    path = _local_parser_path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".docx":
        from app.services.parser.word_parser import parse_word_document

        items = parse_word_document(path, image_analysis_workers=image_analysis_workers)
    elif suffix == ".xlsx":
        from app.services.parser.excel_parser import parse_excel_document

        items = parse_excel_document(path, image_analysis_workers=image_analysis_workers)
    elif suffix == ".pptx":
        from app.services.parser.powerpoint_parser import parse_powerpoint_document

        items = parse_powerpoint_document(path, image_analysis_workers=image_analysis_workers)
    elif suffix == ".pdf":
        if is_standard_pdf_source(source):
            from app.services.parser.standard_pdf_parser import parse_standard_pdf_sections
            from app.services.parser.standard_pdf_splitter import split_standard_pdf_document

            sections = split_standard_pdf_document(
                path,
                image_analysis_workers=image_analysis_workers,
                source_reference=source,
            )
            items = parse_standard_pdf_sections(sections)
        else:
            from app.services.parser.pdf_parser import parse_pdf_document

            items = parse_pdf_document(path, image_analysis_workers=image_analysis_workers)
    else:
        raise ValueError(f"Unsupported document type: {suffix or 'unknown'}")

    # `path` may be a cache file while `source` is a MinIO URI.  Synchronize the
    # parser's real output to the source-derived local path and the archive.
    synchronize_processed_assets(
        source,
        produced_processing_dir=processing_document_dir(path),
    )
    return items


def _local_parser_path(file_path: str | Path) -> Path:
    source = str(file_path)
    if _is_minio_document_source(source):
        return download_raw_document_to_file(parse_raw_document_reference(source).uri)

    path = Path(source)
    if _is_local_data_raw_path(path):
        return download_raw_document_to_file(source)
    return path


def _is_minio_document_source(value: str) -> bool:
    normalized = value.strip().replace("\\", "/")
    parsed = urlparse(normalized)
    return parsed.scheme in {"minio", "s3"} or "data/raw/" in normalized.lower()


def _is_local_data_raw_path(path: Path) -> bool:
    try:
        path.resolve().relative_to((Path.cwd() / "data" / "raw").resolve())
    except (OSError, ValueError):
        return False
    return True


def is_standard_pdf_source(value: str | Path) -> bool:
    """Return whether a PDF needs the split-then-parse standard pipeline."""
    normalized = str(value).strip().replace("\\", "/")
    decoded = unquote(normalized)
    parsed = urlparse(decoded)
    source_path = Path(parsed.path if parsed.scheme else decoded)
    if source_path.suffix.lower() != ".pdf" or is_generated_standard_pdf_section(decoded):
        return False
    return (
        "标准文档" in decoded
        or "产品标准" in decoded
        or source_path.name.upper().startswith("ASME")
    )


def is_generated_standard_pdf_section(value: str | Path) -> bool:
    """Return whether a path belongs to generated ``(切分版)`` section PDFs."""
    return "(切分版)" in unquote(str(value).replace("\\", "/"))


# Backward-compatible alias for callers that used the previous private helper.
_is_standard_pdf_source = is_standard_pdf_source


if __name__ == "__main__":
    target_file = "minio://knowledge-raw-docs/process_guide/订阅号运营SOP.docx"
    parsed_items = parse_document(target_file)
    document_name = Path(target_file).stem
    txt_path = Path("data") / "processing" / document_name / "txt" / f"{document_name}.txt"

    print(f"File: {target_file}")
    print(f"Parsed text blocks: {len(parsed_items)}")
    print(f"Text output: {txt_path}")
