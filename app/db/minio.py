from __future__ import annotations

import mimetypes
import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlparse

from minio import Minio
from minio.error import S3Error

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
RAW_DOCUMENT_CACHE_ROOT = Path("data") / "processing" / ".minio_cache"
LOCAL_RAW_ROOT = Path("data") / "raw"


@dataclass(frozen=True)
class MinioUploadResult:
    bucket: str
    object_name: str
    etag: str | None
    version_id: str | None
    content_type: str
    size: int
    url: str


@dataclass(frozen=True)
class RawDocumentObject:
    bucket: str
    object_name: str

    @property
    def uri(self) -> str:
        return build_minio_uri(self.bucket, self.object_name)

    @property
    def url(self) -> str:
        return build_object_url(self.bucket, self.object_name)


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
    raw_root: str | Path | None = None,
) -> MinioUploadResult:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Upload source is not a file: {path}")

    target_object_name = object_name or raw_document_object_name_for_file(
        path,
        category=category,
        raw_root=raw_root,
    )
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


def build_minio_uri(bucket: str, object_name: str) -> str:
    return f"minio://{bucket}/{quote(object_name, safe='/')}"


def raw_document_object_name_for_file(
    file_path: str | Path,
    *,
    category: str | None = None,
    raw_root: str | Path | None = None,
) -> str:
    path = Path(file_path)
    if category:
        return _normalize_object_name(f"{resolve_raw_document_prefix(category)}/{path.name}")

    root = Path(raw_root) if raw_root is not None else LOCAL_RAW_ROOT
    try:
        relative = path.resolve().relative_to((Path.cwd() / root).resolve())
    except ValueError:
        try:
            relative = path.resolve().relative_to(root.resolve())
        except ValueError:
            return _normalize_object_name(path.name)

    return _normalize_object_name(relative.as_posix())


def parse_raw_document_reference(
    raw_path: str | Path,
    *,
    bucket: str | None = None,
) -> RawDocumentObject:
    text = unquote(str(raw_path)).strip().strip("\"'").replace("\\", "/")
    if not text:
        raise ValueError("Missing raw document path")

    target_bucket = bucket or settings.minio_bucket or DEFAULT_RAW_DOCUMENT_BUCKET
    parsed = urlparse(text)

    if parsed.scheme in {"minio", "s3"}:
        if not parsed.netloc:
            raise ValueError(f"Missing bucket in MinIO document reference: {raw_path}")
        return RawDocumentObject(parsed.netloc, _normalize_object_name(parsed.path.lstrip("/")))

    if parsed.scheme in {"http", "https"}:
        public_ref = _parse_public_object_url(text)
        if public_ref is not None:
            return public_ref

    return RawDocumentObject(target_bucket, _normalize_object_name(_strip_local_raw_prefix(text, target_bucket)))


def raw_document_object_exists(raw_path: str | Path, *, bucket: str | None = None) -> bool:
    reference = parse_raw_document_reference(raw_path, bucket=bucket)
    try:
        get_minio_client().stat_object(reference.bucket, reference.object_name)
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchBucket", "NoSuchObject"}:
            return False
        raise
    return True


def download_raw_document_to_file(
    raw_path: str | Path,
    *,
    bucket: str | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    reference = parse_raw_document_reference(raw_path, bucket=bucket)
    cache_path = cached_raw_document_path(reference, cache_root=cache_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    get_minio_client().fget_object(reference.bucket, reference.object_name, str(cache_path))
    return cache_path


def cached_raw_document_path(
    reference: RawDocumentObject,
    *,
    cache_root: str | Path | None = None,
) -> Path:
    root = Path(cache_root) if cache_root is not None else RAW_DOCUMENT_CACHE_ROOT
    digest = hashlib.sha1(reference.object_name.encode("utf-8")).hexdigest()[:16]
    filename = PurePosixPath(reference.object_name).name or digest
    return root / reference.bucket / digest / filename


def list_raw_document_objects(
    prefix: str | Path = "",
    *,
    bucket: str | None = None,
    recursive: bool = True,
) -> list[RawDocumentObject]:
    if str(prefix or "").strip():
        reference = parse_raw_document_reference(str(prefix), bucket=bucket)
    else:
        reference = RawDocumentObject(bucket or settings.minio_bucket or DEFAULT_RAW_DOCUMENT_BUCKET, "")
    object_prefix = reference.object_name.strip("/")
    if object_prefix and not object_prefix.endswith("/"):
        try:
            get_minio_client().stat_object(reference.bucket, object_prefix)
        except S3Error as exc:
            if exc.code not in {"NoSuchKey", "NoSuchObject"}:
                raise
        else:
            return [RawDocumentObject(reference.bucket, object_prefix)]
        object_prefix = f"{object_prefix}/"

    objects = get_minio_client().list_objects(
        reference.bucket,
        prefix=object_prefix,
        recursive=recursive,
    )
    return [
        RawDocumentObject(reference.bucket, item.object_name)
        for item in objects
        if item.object_name and not item.object_name.endswith("/.keep")
    ]


def _parse_public_object_url(url: str) -> RawDocumentObject | None:
    parsed = urlparse(url)
    path_parts = [part for part in unquote(parsed.path).split("/") if part]
    if not path_parts:
        return None

    configured_public = urlparse(settings.minio_public_endpoint)
    configured_endpoint = urlparse(settings.minio_endpoint)
    known_hosts = {
        parsed_endpoint.netloc.lower()
        for parsed_endpoint in (configured_public, configured_endpoint)
        if parsed_endpoint.netloc
    }
    default_bucket = settings.minio_bucket or DEFAULT_RAW_DOCUMENT_BUCKET

    if parsed.netloc.lower() in known_hosts and path_parts[0] == default_bucket:
        return RawDocumentObject(default_bucket, _normalize_object_name("/".join(path_parts[1:])))
    return None


def _strip_local_raw_prefix(path: str, bucket: str) -> str:
    normalized = path.strip().strip("/")
    lower_path = normalized.lower()
    for marker in ("data/raw/", "data\\raw\\"):
        marker_index = lower_path.find(marker.replace("\\", "/"))
        if marker_index >= 0:
            return normalized[marker_index + len(marker.replace("\\", "/")) :]
    bucket_prefix = f"{bucket}/"
    if lower_path.startswith(bucket_prefix.lower()):
        return normalized[len(bucket_prefix) :]
    return normalized


def _normalize_object_name(object_name: str) -> str:
    parts = []
    for part in unquote(object_name).replace("\\", "/").split("/"):
        part = part.strip()
        if not part or part == ".":
            continue
        if part == "..":
            raise ValueError(f"Invalid MinIO object path: {object_name}")
        parts.append(part)
    normalized = "/".join(parts)
    if not normalized:
        raise ValueError("Missing MinIO object name")
    return normalized


def _normalize_endpoint(endpoint: str) -> tuple[str, bool]:
    parsed = urlparse(endpoint)
    if parsed.scheme:
        return parsed.netloc, parsed.scheme == "https"
    return endpoint, settings.minio_secure


class _EmptyReader:
    def read(self, size: int = -1) -> bytes:
        return b""
