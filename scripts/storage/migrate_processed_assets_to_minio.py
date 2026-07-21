"""Archive legacy parsed ``txt`` and ``img`` files to MinIO.

This is deliberately an opt-in migration: inspect the JSON report first, then
run with ``--apply``.  It never deletes local files, chunks, embeddings, or raw
objects.  A same-stem collision such as ``manual.docx`` and ``manual.pdf`` is
reported and skipped because legacy local processing folders cannot distinguish
their assets safely.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.minio import (  # noqa: E402
    ensure_bucket,
    get_minio_client,
    guess_content_type,
    list_raw_document_objects,
)
from app.services.processed_document_assets import (  # noqa: E402
    ARCHIVED_SUBDIRECTORIES,
    processed_document_bucket,
    processed_document_prefix,
    processed_document_uri,
    set_registry_processed_document_path,
)


SUPPORTED_EXTENSIONS = {".docx", ".xlsx", ".pptx", ".pdf"}


@dataclass(frozen=True)
class MigrationCandidate:
    raw_uri: str
    raw_object_name: str
    processing_dir: Path


def index_processing_directories(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if path.is_dir():
            index.setdefault(path.name, []).append(path)
    return index


def select_processing_dir(
    object_name: str,
    *,
    processing_root: Path,
    name_index: dict[str, list[Path]],
) -> tuple[Path | None, str | None]:
    raw_path = PurePosixPath(object_name)
    expected = processing_root.joinpath(*raw_path.with_suffix("").parts)
    if expected.is_dir():
        return expected, None

    matches = name_index.get(raw_path.stem, [])
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, "processing folder not found"
    return None, f"ambiguous processing folders: {[str(path) for path in matches]}"


def asset_files(processing_dir: Path) -> list[Path]:
    return [
        path
        for path in processing_dir.rglob("*")
        if path.is_file()
        and any(part in ARCHIVED_SUBDIRECTORIES for part in path.relative_to(processing_dir).parts)
    ]


def upload_assets(candidate: MigrationCandidate, *, bucket: str) -> int:
    client = get_minio_client()
    prefix = processed_document_prefix(candidate.raw_uri)
    count = 0
    for asset in asset_files(candidate.processing_dir):
        object_name = f"{prefix}/{asset.relative_to(candidate.processing_dir).as_posix()}"
        client.fput_object(bucket, object_name, str(asset), content_type=guess_content_type(asset.name))
        count += 1
    return count


def build_candidates(processing_root: Path) -> tuple[list[MigrationCandidate], list[dict[str, str]]]:
    raw_objects = [
        item
        for item in list_raw_document_objects("", recursive=True)
        if PurePosixPath(item.object_name).suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    # Legacy processing folders drop the extension, so these cannot be migrated
    # with confidence even when only one local directory is found.
    stem_counts = Counter(str(PurePosixPath(item.object_name).with_suffix("")) for item in raw_objects)
    name_index = index_processing_directories(processing_root)
    candidates: list[MigrationCandidate] = []
    skipped: list[dict[str, str]] = []
    for raw in raw_objects:
        key = str(PurePosixPath(raw.object_name).with_suffix(""))
        if stem_counts[key] > 1:
            skipped.append({"raw_uri": raw.uri, "reason": "same-stem raw document collision"})
            continue
        directory, reason = select_processing_dir(
            raw.object_name,
            processing_root=processing_root,
            name_index=name_index,
        )
        if directory is None:
            skipped.append({"raw_uri": raw.uri, "reason": reason or "unknown"})
            continue
        if not asset_files(directory):
            skipped.append({"raw_uri": raw.uri, "reason": "no txt or img assets"})
            continue
        candidates.append(MigrationCandidate(raw.uri, raw.object_name, directory))
    return candidates, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processing-root", type=Path, default=PROJECT_ROOT / "data" / "processing")
    parser.add_argument("--bucket", default=processed_document_bucket())
    parser.add_argument("--apply", action="store_true", help="Upload assets and update PostgreSQL. Default is report-only.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    candidates, skipped = build_candidates(args.processing_root)
    report: dict[str, object] = {
        "dry_run": not args.apply,
        "processing_root": str(args.processing_root),
        "processed_bucket": args.bucket,
        "candidate_count": len(candidates),
        "skipped": skipped,
    }
    if not args.apply:
        report["candidates"] = [
            {
                "raw_uri": candidate.raw_uri,
                "processing_dir": str(candidate.processing_dir),
                "processed_document_path": processed_document_uri(candidate.raw_uri, bucket=args.bucket),
                "asset_count": len(asset_files(candidate.processing_dir)),
            }
            for candidate in candidates
        ]
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    ensure_bucket(args.bucket)
    migrated: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            uploaded = upload_assets(candidate, bucket=args.bucket)
            uri = processed_document_uri(candidate.raw_uri, bucket=args.bucket)
            set_registry_processed_document_path(candidate.raw_uri, uri)
            migrated.append({"raw_uri": candidate.raw_uri, "uploaded": uploaded, "processed_document_path": uri})
        except Exception as exc:
            failures.append({"raw_uri": candidate.raw_uri, "error": str(exc)})
            if not args.continue_on_error:
                break
    report["migrated"] = migrated
    report["failures"] = failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
