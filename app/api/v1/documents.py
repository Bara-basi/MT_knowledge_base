from __future__ import annotations

import mimetypes
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.db.minio import (
    RAW_DOCUMENT_CATEGORIES,
    ensure_bucket,
    ensure_raw_document_prefixes,
    upload_raw_document_stream,
)

router = APIRouter(prefix="/documents", tags=["documents"])
_raw_document_root = Path("data") / "raw"


@router.get("/minio/categories")
def list_minio_categories() -> dict[str, object]:
    return {
        "categories": RAW_DOCUMENT_CATEGORIES,
        "prefixes": sorted(RAW_DOCUMENT_CATEGORIES.values()),
    }


@router.get("/download")
def download_raw_document(path: str) -> FileResponse:
    try:
        document_path = resolve_raw_document_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    media_type = mimetypes.guess_type(document_path.name)[0] or "application/octet-stream"
    return FileResponse(
        document_path,
        media_type=media_type,
        filename=document_path.name,
    )


def resolve_raw_document_path(raw_path: str, raw_root: Path | None = None) -> Path:
    root = (raw_root or _raw_document_root).resolve()
    normalized_path = _normalize_requested_document_path(raw_path)
    if not normalized_path:
        raise ValueError("Missing document path")

    for candidate in _iter_candidate_document_paths(normalized_path, root):
        if candidate.exists() and candidate.is_file() and _is_relative_to(candidate, root):
            return candidate

    filename = PurePosixPath(normalized_path).name
    if not filename:
        raise FileNotFoundError(f"Document not found: {raw_path}")

    exact_matches = [path for path in root.rglob(filename) if path.is_file()]
    if exact_matches:
        return exact_matches[0]

    lower_filename = filename.lower()
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() == lower_filename:
            return path

    raise FileNotFoundError(f"Document not found: {raw_path}")


def _normalize_requested_document_path(raw_path: str) -> str:
    return unquote(raw_path).strip().strip("\"'").replace("\\", "/")


def _iter_candidate_document_paths(normalized_path: str, root: Path) -> list[Path]:
    candidates: list[Path] = []
    lower_path = normalized_path.lower()
    data_raw_marker = "data/raw/"

    if data_raw_marker in lower_path:
        marker_index = lower_path.index(data_raw_marker)
        relative_to_raw = normalized_path[marker_index + len(data_raw_marker) :]
        candidates.append((root / relative_to_raw).resolve())
        candidates.append(Path(normalized_path[marker_index:]).resolve())

    direct_path = Path(normalized_path)
    candidates.append(direct_path.resolve())
    candidates.append((root / normalized_path).resolve())

    return candidates


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


@router.post("/minio/init")
def init_minio_raw_document_folders(
    bucket: str | None = None,
) -> dict[str, object]:
    try:
        target_bucket = ensure_bucket(bucket=bucket)
        prefixes = ensure_raw_document_prefixes(bucket=target_bucket)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"bucket": target_bucket, "prefixes": prefixes}


@router.post("/minio/upload")
async def upload_document_to_minio(
    file: UploadFile = File(...),
    category: str | None = Form(default=None),
    bucket: str | None = Form(default=None),
    object_name: str | None = Form(default=None),
) -> dict[str, object]:
    try:
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
        result = upload_raw_document_stream(
            data=file.file,
            filename=file.filename or "uploaded-file",
            size=size,
            category=category,
            bucket=bucket,
            object_name=object_name,
            content_type=file.content_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await file.close()

    return {
        "bucket": result.bucket,
        "object_name": result.object_name,
        "etag": result.etag,
        "version_id": result.version_id,
        "content_type": result.content_type,
        "size": result.size,
        "url": result.url,
    }
