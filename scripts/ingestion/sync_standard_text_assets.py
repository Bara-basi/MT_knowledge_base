from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minio.error import S3Error

from app.core.config import settings
from app.db.minio import DEFAULT_STANDARD_ASSET_BUCKET, ensure_bucket, get_minio_client


DEFAULT_SOURCE_ROOT = Path("data") / "processing" / "产品标准"


def standard_text_object_name(path: Path, source_root: Path) -> str:
    relative = path.resolve().relative_to(source_root.resolve())
    if len(relative.parts) < 4 or relative.parts[-2].casefold() != "txt":
        raise ValueError(f"Unexpected parsed standard TXT layout: {path}")
    volume, document_name = relative.parts[0], relative.parts[1]
    volume_asset_name = volume if volume.endswith("(切分版)") else f"{volume}(切分版)"
    return str(
        PurePosixPath("产品标准")
        / volume_asset_name
        / document_name
        / relative.name
    )


def sync_standard_text_assets(
    source_root: Path,
    *,
    bucket: str,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, int]:
    files = sorted(source_root.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No parsed standard TXT files found under {source_root}")

    client = get_minio_client()
    if not dry_run:
        ensure_bucket(bucket)

    uploaded = 0
    skipped = 0
    for path in files:
        object_name = standard_text_object_name(path, source_root)
        if not force and not dry_run:
            try:
                stat = client.stat_object(bucket, object_name)
            except S3Error as exc:
                if exc.code not in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
                    raise
            else:
                if int(getattr(stat, "size", -1)) == path.stat().st_size:
                    skipped += 1
                    continue
        if dry_run:
            print(f"DRY-RUN {path} -> minio://{bucket}/{object_name}")
            continue
        client.fput_object(
            bucket,
            object_name,
            str(path),
            content_type="text/plain; charset=utf-8",
        )
        uploaded += 1

    return {"found": len(files), "uploaded": uploaded, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync parsed product-standard TXT files to MinIO.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--bucket",
        default=settings.minio_standard_asset_bucket or DEFAULT_STANDARD_ASSET_BUCKET,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = sync_standard_text_assets(
        args.source_root,
        bucket=args.bucket,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(result)


if __name__ == "__main__":
    main()
