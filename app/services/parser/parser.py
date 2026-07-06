from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

try:
    from app.db.minio import download_raw_document_to_file, parse_raw_document_reference
    from app.services.parser.excel_parser import parse_excel_document
    from app.services.parser.pdf_parser import parse_pdf_document
    from app.services.parser.powerpoint_parser import parse_powerpoint_document
    from app.services.parser.standard_pdf_parser import parse_standard_pdf_document
    from app.services.parser.word_parser import parse_word_document
except ModuleNotFoundError:
    from app.db.minio import download_raw_document_to_file, parse_raw_document_reference
    from excel_parser import parse_excel_document
    from pdf_parser import parse_pdf_document
    from powerpoint_parser import parse_powerpoint_document
    from standard_pdf_parser import parse_standard_pdf_document
    from word_parser import parse_word_document


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
        return parse_word_document(path, image_analysis_workers=image_analysis_workers)
    if suffix == ".xlsx":
        return parse_excel_document(path, image_analysis_workers=image_analysis_workers)
    if suffix == ".pptx":
        return parse_powerpoint_document(path, image_analysis_workers=image_analysis_workers)
    if suffix == ".pdf":
        if _is_standard_pdf_source(source):
            return parse_standard_pdf_document(
                path,
                image_analysis_workers=image_analysis_workers,
                source_reference=source,
            )
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
