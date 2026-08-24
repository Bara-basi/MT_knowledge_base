from __future__ import annotations

import asyncio
import hmac
import mimetypes
from pathlib import PurePosixPath
from urllib.parse import quote

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from minio.error import S3Error

from app.core.config import settings
from app.db.minio import (
    RAW_DOCUMENT_CATEGORIES,
    RawDocumentObject,
    ensure_bucket,
    ensure_raw_document_prefixes,
    get_minio_client,
    parse_raw_document_reference,
    upload_raw_document_stream,
)
from app.services.lark_document_sync import ingest_lark_document_link, scan_lark_updates
from app.services.agent_documents import (
    MAX_AGENT_CONTEXT_DOCUMENTS,
    MAX_AGENT_CONTEXT_IMAGES,
    MAX_AGENT_PDF_PAGES,
    is_missing_minio_object,
    prepare_agent_context_assets,
    prepare_agent_vision_payload,
)
from app.services.marketing_asset_catalog import (
    format_harness_results,
    search_marketing_assets,
)

router = APIRouter(prefix="/documents", tags=["documents"])


class AgentVisionPayloadRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=4096)
    max_pages: int = Field(
        MAX_AGENT_PDF_PAGES,
        ge=1,
        le=MAX_AGENT_PDF_PAGES,
    )


class AgentContextAssetsRequest(BaseModel):
    document_paths: list[str] = Field(default_factory=list, max_length=MAX_AGENT_CONTEXT_DOCUMENTS)
    image_paths: list[str] = Field(default_factory=list, max_length=MAX_AGENT_CONTEXT_IMAGES)


class MarketingAssetSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Marketing asset name or path keywords.")
    limit: int = Field(10, ge=1, le=20, description="Maximum matching marketing assets.")
    source: str = Field(
        "api",
        min_length=1,
        max_length=64,
        description="Use harness to receive a compact model-readable result; other values receive the standard response.",
    )


class HarnessAttachmentRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=256)
    internal_session_id: str = Field(..., min_length=1, max_length=128)
    attachment_id: str = Field(..., min_length=1, max_length=128)


class HarnessAttachmentReadRequest(HarnessAttachmentRequest):
    query: str | None = Field(None, max_length=500)
    offset: int = Field(0, ge=0)
    limit: int = Field(3, ge=1, le=4)


def _verify_attachment_api_access(request: Request, token: str | None) -> None:
    expected = settings.harness_attachment_api_token
    if expected:
        if not token or not hmac.compare_digest(token, expected):
            raise HTTPException(status_code=401, detail="Invalid internal attachment token")
        return
    host = (request.client.host if request.client else "").strip().lower()
    if host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(
            status_code=403,
            detail="Attachment API is local-only until HARNESS_ATTACHMENT_API_TOKEN is configured",
        )


@router.post("/harness-attachments/upload")
async def upload_harness_attachment(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    internal_session_id: str = Form(...),
    x_kb_internal_token: str | None = Header(None),
) -> dict[str, object]:
    """Store a model attachment without exposing its local path."""
    _verify_attachment_api_access(request, x_kb_internal_token)
    from app.services.harness_attachments import save_attachment

    content = await file.read(settings.harness_attachment_max_bytes + 1)
    try:
        return save_attachment(
            user_id=user_id,
            internal_session_id=internal_session_id,
            filename=file.filename or "attachment",
            content=content,
            content_type=file.content_type,
            source="api",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()


@router.get("/harness-attachments")
def list_harness_attachments(
    request: Request,
    user_id: str,
    internal_session_id: str,
    x_kb_internal_token: str | None = Header(None),
) -> dict[str, object]:
    _verify_attachment_api_access(request, x_kb_internal_token)
    from app.services.harness_attachments import list_attachments

    return {"attachments": list_attachments(user_id=user_id, internal_session_id=internal_session_id)}


@router.post("/harness-attachments/parse")
async def parse_harness_attachment(
    payload: HarnessAttachmentRequest,
    request: Request,
    x_kb_internal_token: str | None = Header(None),
) -> dict[str, object]:
    _verify_attachment_api_access(request, x_kb_internal_token)
    from app.services.harness_attachments import parse_attachment

    try:
        return await asyncio.to_thread(
            parse_attachment,
            user_id=payload.user_id,
            internal_session_id=payload.internal_session_id,
            attachment_id=payload.attachment_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/harness-attachments/read")
async def read_harness_attachment(
    payload: HarnessAttachmentReadRequest,
    request: Request,
    x_kb_internal_token: str | None = Header(None),
) -> dict[str, object]:
    _verify_attachment_api_access(request, x_kb_internal_token)
    from app.services.harness_attachments import read_attachment

    try:
        return await asyncio.to_thread(
            read_attachment,
            user_id=payload.user_id,
            internal_session_id=payload.internal_session_id,
            attachment_id=payload.attachment_id,
            query=payload.query,
            offset=payload.offset,
            limit=payload.limit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/minio/categories")
def list_minio_categories() -> dict[str, object]:
    return {
        "categories": RAW_DOCUMENT_CATEGORIES,
        "prefixes": sorted(RAW_DOCUMENT_CATEGORIES.values()),
    }


@router.post("/marketing-assets/search")
def search_marketing_asset_catalog(request: MarketingAssetSearchRequest) -> dict[str, object]:
    """Find marketing materials by catalogue path without retrieving document content."""
    try:
        matches = search_marketing_assets(request.query, limit=request.limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Marketing asset search failed: {exc}") from exc

    if request.source == "harness":
        return {
            "query": request.query,
            "source": "harness",
            "count": len(matches),
            "text": format_harness_results(request.query, matches),
        }
    return {
        "query": request.query,
            "source": request.source,
        "count": len(matches),
        "matches": [match.to_dict() for match in matches],
    }


@router.get("/download")
def download_raw_document(path: str) -> StreamingResponse:
    try:
        reference = resolve_raw_document_object(path)
        client = get_minio_client()
        stat = client.stat_object(reference.bucket, reference.object_name)
        response = client.get_object(reference.bucket, reference.object_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchBucket", "NoSuchObject"}:
            raise HTTPException(status_code=404, detail=f"Document not found: {path}") from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    filename = PurePosixPath(reference.object_name).name or "document"
    media_type = (
        getattr(stat, "content_type", None)
        or mimetypes.guess_type(filename)[0]
        or "application/octet-stream"
    )
    headers = {
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
    }
    return StreamingResponse(
        _stream_minio_response(response),
        media_type=media_type,
        headers=headers,
    )


@router.post("/agent-vision-payload")
def prepare_document_for_agent_vision(
    request: AgentVisionPayloadRequest,
) -> dict[str, object]:
    try:
        return prepare_agent_vision_payload(
            request.path,
            max_pages=request.max_pages,
        ).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except S3Error as exc:
        if is_missing_minio_object(exc):
            raise HTTPException(status_code=404, detail=f"Document not found: {request.path}") from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document preparation failed: {exc}") from exc


@router.post("/agent-context-assets")
def prepare_context_assets_for_final_agent(
    request: AgentContextAssetsRequest,
) -> dict[str, object]:
    try:
        return prepare_agent_context_assets(
            request.document_paths,
            request.image_paths,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent asset preparation failed: {exc}") from exc


def resolve_raw_document_object(raw_path: str, bucket: str | None = None) -> RawDocumentObject:
    try:
        return parse_raw_document_reference(raw_path, bucket=bucket)
    except ValueError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _stream_minio_response(response):
    try:
        yield from response.stream(amt=1024 * 1024)
    finally:
        response.close()
        response.release_conn()


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


@router.post("/sync/lark/scan")
def scan_lark_document_updates(
    source: str | None = None,
    dry_run: bool = False,
    image_workers: int = 3,
) -> dict[str, object]:
    try:
        result = scan_lark_updates(
            source=source or "data/src/vector_src.json",
            dry_run=dry_run,
            image_analysis_workers=image_workers,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result.to_dict()


@router.post("/sync/lark/update")
def update_lark_document_by_name(
    document_name: str,
    source: str | None = None,
    force: bool = True,
    dry_run: bool = False,
    image_workers: int = 3,
) -> dict[str, object]:
    try:
        result = scan_lark_updates(
            source=source or "data/src/vector_src.json",
            document_name=document_name,
            force=force,
            dry_run=dry_run,
            image_analysis_workers=image_workers,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result.to_dict()


@router.post("/sync/lark/ingest")
def ingest_lark_document(
    document_link: str,
    bucket: str = "knowledge-raw-docs",
    object_name: str | None = None,
    category: str | None = None,
    source_name: str | None = None,
    image_workers: int = 3,
) -> dict[str, object]:
    try:
        result = ingest_lark_document_link(
            document_link,
            bucket=bucket,
            object_name=object_name,
            category=category,
            source_name=source_name,
            image_analysis_workers=image_workers,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if result.status == "failed":
        raise HTTPException(status_code=500, detail=result.to_dict())
    return result.to_dict()
