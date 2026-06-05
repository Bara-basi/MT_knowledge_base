from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.db.minio import (
    RAW_DOCUMENT_CATEGORIES,
    ensure_bucket,
    ensure_raw_document_prefixes,
    upload_raw_document_stream,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/minio/categories")
def list_minio_categories() -> dict[str, object]:
    return {
        "categories": RAW_DOCUMENT_CATEGORIES,
        "prefixes": sorted(RAW_DOCUMENT_CATEGORIES.values()),
    }


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
