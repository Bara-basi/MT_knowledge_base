from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlparse

try:
    import fitz
except ModuleNotFoundError:  # pragma: no cover - deployment dependency guard
    fitz = None

from minio.error import S3Error

from app.core.config import settings
from app.db.minio import (
    DEFAULT_RAW_DOCUMENT_BUCKET,
    DEFAULT_STANDARD_ASSET_BUCKET,
    RawDocumentObject,
    build_minio_uri,
    get_minio_client,
    parse_raw_document_reference,
)


MAX_AGENT_ASSET_BYTES = 20 * 1024 * 1024
MAX_AGENT_PDF_PAGES = 20
MAX_RENDERED_PAYLOAD_BYTES = 32 * 1024 * 1024
MAX_AGENT_TEXT_BYTES = 512 * 1024
MAX_AGENT_CONTEXT_DOCUMENTS = 2
MAX_AGENT_CONTEXT_IMAGES = 4
PDF_RENDER_SCALE = 1.5
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


@dataclass(frozen=True)
class AgentVisionPayload:
    source_path: str
    media_type: str
    kind: str
    vision_inputs: list[dict[str, object]]
    size: int
    page_count: int | None = None
    truncated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "media_type": self.media_type,
            "kind": self.kind,
            "vision_inputs": self.vision_inputs,
            "size": self.size,
            "page_count": self.page_count,
            "truncated": self.truncated,
            "citation_allowed": True,
        }


@dataclass(frozen=True)
class AgentTextPayload:
    source_path: str
    asset_path: str
    content: str
    size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "asset_path": self.asset_path,
            "media_type": "text/plain",
            "kind": "parsed_text",
            "content": self.content,
            "size": self.size,
            "citation_allowed": True,
        }


def resolve_agent_document_reference(raw_path: str) -> RawDocumentObject:
    """Resolve only explicit MinIO references from the two knowledge buckets."""

    text = str(raw_path or "").strip()
    parsed = urlparse(text)
    if parsed.scheme not in {"minio", "s3"}:
        raise ValueError("Agent document path must be an explicit minio:// or s3:// reference")

    reference = parse_raw_document_reference(text)
    allowed_buckets = {
        settings.minio_bucket or DEFAULT_RAW_DOCUMENT_BUCKET,
        settings.minio_standard_asset_bucket or DEFAULT_STANDARD_ASSET_BUCKET,
    }
    if reference.bucket not in allowed_buckets:
        raise ValueError(f"MinIO bucket is not available to the QA agent: {reference.bucket}")
    return reference


def prepare_agent_vision_payload(
    path: str,
    *,
    max_pages: int = MAX_AGENT_PDF_PAGES,
) -> AgentVisionPayload:
    """Prepare image inputs for a Kimi call made by n8n; no model runs here."""

    reference = resolve_agent_document_reference(path)
    data, media_type = _download_asset(reference)
    max_pages = max(1, min(int(max_pages), MAX_AGENT_PDF_PAGES))

    if media_type == "application/pdf":
        return _render_pdf_payload(
            data,
            source_path=reference.uri,
            max_pages=max_pages,
        )
    if media_type in SUPPORTED_IMAGE_TYPES:
        return AgentVisionPayload(
            source_path=reference.uri,
            media_type=media_type,
            kind="image",
            vision_inputs=[_vision_input(data, media_type, page_number=None)],
            size=len(data),
        )
    raise ValueError(f"Unsupported agent document media type: {media_type}")


def prepare_agent_text_payload(path: str) -> AgentTextPayload:
    """Read the parsed TXT matching a recalled standard document from MinIO."""

    source = resolve_agent_document_reference(path)
    text_reference = resolve_standard_text_reference(source)
    data, media_type = _download_asset(text_reference, max_bytes=MAX_AGENT_TEXT_BYTES)
    if media_type not in {"text/plain", "application/octet-stream"} and not text_reference.object_name.lower().endswith(".txt"):
        raise ValueError(f"Resolved standard asset is not TXT: {text_reference.uri}")
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Parsed standard TXT is not UTF-8: {text_reference.uri}") from exc
    if not content.strip():
        raise ValueError(f"Parsed standard TXT is empty: {text_reference.uri}")
    return AgentTextPayload(
        source_path=source.uri,
        asset_path=text_reference.uri,
        content=content,
        size=len(data),
    )


def prepare_agent_context_assets(
    document_paths: list[str],
    image_paths: list[str],
) -> dict[str, object]:
    """Prepare deterministic text and image inputs for the final n8n model call."""

    documents: list[dict[str, object]] = []
    images: list[dict[str, object]] = []
    warnings: list[dict[str, str]] = []

    for path in _dedupe_paths(document_paths)[:MAX_AGENT_CONTEXT_DOCUMENTS]:
        try:
            documents.append(prepare_agent_text_payload(path).to_dict())
        except (ValueError, S3Error) as exc:
            warnings.append({"path": path, "reason": str(exc)})

    for path in _dedupe_paths(image_paths)[:MAX_AGENT_CONTEXT_IMAGES]:
        try:
            payload = prepare_agent_vision_payload(path)
            if payload.kind != "image" or len(payload.vision_inputs) != 1:
                raise ValueError("Only direct image assets may be attached to the final Agent")
            data_url = str(payload.vision_inputs[0]["image_url"]["url"])
            _, encoded = data_url.split(",", 1)
            images.append(
                {
                    "source_path": payload.source_path,
                    "media_type": payload.media_type,
                    "kind": "image",
                    "data_base64": encoded,
                    "size": payload.size,
                }
            )
        except (ValueError, S3Error) as exc:
            warnings.append({"path": path, "reason": str(exc)})

    return {
        "documents": documents,
        "images": images,
        "warnings": warnings,
        "limits": {
            "documents": MAX_AGENT_CONTEXT_DOCUMENTS,
            "images": MAX_AGENT_CONTEXT_IMAGES,
        },
    }


def resolve_standard_text_reference(source: RawDocumentObject) -> RawDocumentObject:
    """Resolve a same-basename TXT in knowledge-standard-assets.

    The fast path follows the standard asset layout. A recursive exact-filename
    search is retained for legacy objects whose directory layout differs.
    """

    asset_bucket = settings.minio_standard_asset_bucket or DEFAULT_STANDARD_ASSET_BUCKET
    source_path = PurePosixPath(source.object_name)
    stem = source_path.stem
    if not stem:
        raise ValueError(f"Cannot derive a standard filename from: {source.uri}")

    if source.bucket == asset_bucket and source_path.suffix.lower() == ".txt":
        return source

    direct_object = str(source_path.parent / stem / f"{stem}.txt")
    direct = RawDocumentObject(asset_bucket, direct_object)
    client = get_minio_client()
    try:
        client.stat_object(asset_bucket, direct_object)
    except S3Error as exc:
        if not is_missing_minio_object(exc):
            raise
    else:
        return direct

    target_name = f"{stem}.txt".casefold()
    candidates = [
        RawDocumentObject(asset_bucket, item.object_name)
        for item in client.list_objects(asset_bucket, recursive=True)
        if item.object_name
        and PurePosixPath(item.object_name).name.casefold() == target_name
    ]
    if not candidates:
        raise ValueError(f"No parsed TXT found for standard document: {source.uri}")
    if len(candidates) == 1:
        return candidates[0]

    source_parts = [part.casefold() for part in source_path.parts[:-1]]

    def score(reference: RawDocumentObject) -> int:
        candidate_parts = [part.casefold() for part in PurePosixPath(reference.object_name).parts[:-1]]
        return len(set(source_parts) & set(candidate_parts))

    ranked = sorted(candidates, key=lambda item: (-score(item), item.object_name))
    if len(ranked) > 1 and score(ranked[0]) == score(ranked[1]):
        matches = ", ".join(build_minio_uri(item.bucket, item.object_name) for item in ranked[:3])
        raise ValueError(f"Parsed TXT lookup is ambiguous for {source.uri}: {matches}")
    return ranked[0]


def _download_asset(
    reference: RawDocumentObject,
    *,
    max_bytes: int = MAX_AGENT_ASSET_BYTES,
) -> tuple[bytes, str]:
    client = get_minio_client()
    stat = client.stat_object(reference.bucket, reference.object_name)
    size = int(getattr(stat, "size", 0) or 0)
    if size <= 0:
        raise ValueError("MinIO object is empty")
    if size > max_bytes:
        raise ValueError(
            f"MinIO object is too large for the QA agent: {size} bytes "
            f"(maximum {max_bytes})"
        )

    filename = PurePosixPath(reference.object_name).name
    media_type = str(getattr(stat, "content_type", "") or "").split(";", 1)[0].strip().lower()
    guessed_type = mimetypes.guess_type(filename)[0]
    if not media_type or media_type == "application/octet-stream":
        media_type = guessed_type or "application/octet-stream"

    response = client.get_object(reference.bucket, reference.object_name)
    try:
        data = response.read(max_bytes + 1)
    finally:
        response.close()
        response.release_conn()
    if len(data) > max_bytes:
        raise ValueError(f"MinIO object exceeds the {max_bytes}-byte agent limit")
    return data, media_type


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in paths:
        path = str(value or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def _render_pdf_payload(
    data: bytes,
    *,
    source_path: str,
    max_pages: int,
) -> AgentVisionPayload:
    if fitz is None:
        raise ModuleNotFoundError("pymupdf is required to prepare PDF pages for Kimi vision")

    vision_inputs: list[dict[str, object]] = []
    rendered_bytes = 0
    with fitz.open(stream=data, filetype="pdf") as document:
        page_count = int(document.page_count)
        for index in range(min(page_count, max_pages)):
            pixmap = document[index].get_pixmap(
                matrix=fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE),
                alpha=False,
            )
            page_data = pixmap.tobytes("png")
            rendered_bytes += len(page_data)
            if rendered_bytes > MAX_RENDERED_PAYLOAD_BYTES:
                break
            vision_inputs.append(_vision_input(page_data, "image/png", page_number=index + 1))

    if not vision_inputs:
        raise ValueError("PDF could not be rendered within the agent payload limit")
    return AgentVisionPayload(
        source_path=source_path,
        media_type="application/pdf",
        kind="pdf_pages",
        vision_inputs=vision_inputs,
        size=len(data),
        page_count=page_count,
        truncated=len(vision_inputs) < page_count,
    )


def _vision_input(
    data: bytes,
    media_type: str,
    *,
    page_number: int | None,
) -> dict[str, object]:
    encoded = base64.b64encode(data).decode("ascii")
    payload: dict[str, object] = {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
    }
    if page_number is not None:
        payload["page_number"] = page_number
    return payload


def is_missing_minio_object(exc: S3Error) -> bool:
    return exc.code in {"NoSuchKey", "NoSuchBucket", "NoSuchObject"}
