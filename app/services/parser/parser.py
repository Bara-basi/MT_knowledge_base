from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from app.db.minio import download_raw_document_to_file, parse_raw_document_reference


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

        return parse_word_document(path, image_analysis_workers=image_analysis_workers)
    if suffix == ".xlsx":
        from app.services.parser.excel_parser import parse_excel_document

        return parse_excel_document(path, image_analysis_workers=image_analysis_workers)
    if suffix == ".pptx":
        from app.services.parser.powerpoint_parser import parse_powerpoint_document

        return parse_powerpoint_document(path, image_analysis_workers=image_analysis_workers)
    if suffix == ".pdf":
        if _is_standard_pdf_source(source):
            from app.services.parser.standard_pdf_parser import parse_standard_pdf_document

            return parse_standard_pdf_document(
                path,
                image_analysis_workers=image_analysis_workers,
                source_reference=source,
            )
        from app.services.parser.pdf_parser import parse_pdf_document

        return parse_pdf_document(path, image_analysis_workers=image_analysis_workers)

    raise ValueError(f"Unsupported document type: {suffix or 'unknown'}")


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


def _is_standard_pdf_source(value: str) -> bool:
    normalized = value.strip().replace("\\", "/")
    return "标准文档" in normalized and "(切分版)" not in normalized


if __name__ == "__main__":
    target_file = "minio://knowledge-raw-docs/process_guide/订阅号运营SOP.docx"
    parsed_items = parse_document(target_file)
    document_name = Path(target_file).stem
    txt_path = Path("data") / "processing" / document_name / "txt" / f"{document_name}.txt"

    print(f"File: {target_file}")
    print(f"Parsed text blocks: {len(parsed_items)}")
    print(f"Text output: {txt_path}")
