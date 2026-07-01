from __future__ import annotations

from pathlib import Path


RAW_ROOT = Path("data") / "raw"
PROCESSING_ROOT = Path("data") / "processing"


def processing_document_dir(source_file: str | Path) -> Path:
    source_path = Path(source_file)
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
