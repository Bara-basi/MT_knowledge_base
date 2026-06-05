from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlparse

from minio import Minio

from app.core.config import settings


RAW_DOCUMENT_CATEGORIES: dict[str, str] = {
    "\u4e00\u822c\u7c7b\u3001\u6587\u672c\u5c45\u591a\u7684word\u6587\u6863": "common_structure_docx",
    "\u6d41\u7a0b\u7c7b\u6570\u636eWORD": "process_description_docx",
    "\u7ed3\u6784\u5316word\u6587\u6863": "structured_docx",
    "\u4e3b\u9898\u660e\u786e\u7684PPT": "clear_theme_pptx",
    "\u666e\u901a\u8868\u683c": "common_structure_xlsx",
}

RAW_DOCUMENT_PREFIXES = frozenset(RAW_DOCUMENT_CATEGORIES.values())
DEFAULT_RAW_DOCUMENT_BUCKET = "knowledge-raw-docs"


@dataclass(frozen=True)
class MinioUploadResult:
    bucket: str
    object_name: str
    etag: str | None
    version_id: str | None
    content_type: str
    size: int
    url: str


def get_minio_client() -> Minio:
    endpoint, secure = _normalize_endpoint(settings.minio_endpoint)
    return Minio(
        endpoint,
        access_key=settings.minio_access_key_id,
        secret_key=settings.minio_secret_access_key,
        secure=secure,
    )


def ensure_bucket(bucket: str | None = None) -> str:
    target_bucket = bucket or settings.minio_bucket or DEFAULT_RAW_DOCUMENT_BUCKET
    client = get_minio_client()
    if not client.bucket_exists(target_bucket):
        client.make_bucket(target_bucket)
    return target_bucket


def ensure_raw_document_prefixes(bucket: str | None = None) -> list[str]:
    target_bucket = ensure_bucket(bucket)
    client = get_minio_client()
    for prefix in sorted(RAW_DOCUMENT_PREFIXES):
        client.put_object(
            target_bucket,
            f"{prefix}/.keep",
            data=_EmptyReader(),
            length=0,
            content_type="application/octet-stream",
        )
    return sorted(RAW_DOCUMENT_PREFIXES)


def resolve_raw_document_prefix(category: str | None) -> str:
    if not category:
        return "common_structure_docx"
    normalized = category.strip()
    if normalized in RAW_DOCUMENT_PREFIXES:
        return normalized
    if normalized in RAW_DOCUMENT_CATEGORIES:
        return RAW_DOCUMENT_CATEGORIES[normalized]
    allowed = ", ".join(sorted(RAW_DOCUMENT_PREFIXES))
    raise ValueError(f"Unsupported document category: {category}. Allowed: {allowed}")


def upload_raw_document_file(
    file_path: str | Path,
    *,
    category: str | None = None,
    bucket: str | None = None,
    object_name: str | None = None,
) -> MinioUploadResult:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Upload source is not a file: {path}")

    prefix = resolve_raw_document_prefix(category) if category else infer_prefix_from_file(path)
    target_object_name = object_name or f"{prefix}/{path.name}"
    content_type = guess_content_type(path.name)
    target_bucket = ensure_bucket(bucket)
    response = get_minio_client().fput_object(
        target_bucket,
        target_object_name,
        str(path),
        content_type=content_type,
    )
    return MinioUploadResult(
        bucket=target_bucket,
        object_name=target_object_name,
        etag=getattr(response, "etag", None),
        version_id=getattr(response, "version_id", None),
        content_type=content_type,
        size=path.stat().st_size,
        url=build_object_url(target_bucket, target_object_name),
    )


def upload_raw_document_stream(
    data,
    *,
    filename: str,
    size: int,
    category: str | None = None,
    bucket: str | None = None,
    object_name: str | None = None,
    content_type: str | None = None,
) -> MinioUploadResult:
    prefix = resolve_raw_document_prefix(category) if category else infer_prefix_from_name(filename)
    target_object_name = object_name or f"{prefix}/{Path(filename).name}"
    target_content_type = content_type or guess_content_type(filename)
    target_bucket = ensure_bucket(bucket)
    response = get_minio_client().put_object(
        target_bucket,
        target_object_name,
        data=data,
        length=size,
        content_type=target_content_type,
    )
    return MinioUploadResult(
        bucket=target_bucket,
        object_name=target_object_name,
        etag=getattr(response, "etag", None),
        version_id=getattr(response, "version_id", None),
        content_type=target_content_type,
        size=size,
        url=build_object_url(target_bucket, target_object_name),
    )


def infer_prefix_from_file(path: Path) -> str:
    return infer_prefix_from_name(path.name)


def infer_prefix_from_name(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".pptx":
        return "clear_theme_pptx"
    if suffix == ".xlsx":
        return "common_structure_xlsx"
    return "common_structure_docx"


def guess_content_type(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def build_object_url(bucket: str, object_name: str) -> str:
    base_url = settings.minio_public_endpoint.rstrip("/")
    return f"{base_url}/{quote(bucket)}/{quote(object_name, safe='/')}"


def _normalize_endpoint(endpoint: str) -> tuple[str, bool]:
    parsed = urlparse(endpoint)
    if parsed.scheme:
        return parsed.netloc, parsed.scheme == "https"
    return endpoint, settings.minio_secure


class _EmptyReader:
    def read(self, size: int = -1) -> bytes:
        return b""
