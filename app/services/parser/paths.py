from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from app.db.minio import parse_raw_document_reference


RAW_ROOT = Path("data") / "raw"
PROCESSING_ROOT = Path("data") / "processing"


def processing_document_dir(source_file: str | Path) -> Path:
    source_text = str(source_file)
    if _looks_like_minio_source(source_text):
        reference = parse_raw_document_reference(source_text)
        return PROCESSING_ROOT / Path(reference.object_name).with_suffix("")

    source_path = Path(source_text)
    source_abs = _absolute_path(source_path)
    raw_root_abs = (Path.cwd() / RAW_ROOT).resolve()

    try:
        relative = source_abs.relative_to(raw_root_abs)
    except ValueError:
        return PROCESSING_ROOT / source_path.stem

    return PROCESSING_ROOT / relative.with_suffix("")


def processing_subdir(source_file: str | Path, *parts: str) -> Path:
    return processing_document_dir(source_file).joinpath(*parts)


def _absolute_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _looks_like_minio_source(value: str) -> bool:
    normalized = value.strip().replace("\\", "/")
    if not normalized:
        return False
    parsed = urlparse(normalized)
    return parsed.scheme in {"minio", "s3"} or "data/raw/" in normalized.lower()
