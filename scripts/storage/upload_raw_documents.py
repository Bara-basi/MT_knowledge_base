from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_file(PROJECT_ROOT / ".env")

from app.db.minio import (  # noqa: E402
    RAW_DOCUMENT_CATEGORIES,
    ensure_raw_document_prefixes,
    upload_raw_document_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload raw document files to MinIO.",
    )
    parser.add_argument("input_path", help="File or folder to upload.")
    parser.add_argument(
        "--raw-root",
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="Local raw document root used to preserve object paths.",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Target bucket. Defaults to MINIO_BUCKET/APP_MINIO_BUCKET.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help=(
            "Target category prefix or Chinese category name. "
            "When omitted, object names preserve paths relative to --raw-root."
        ),
    )
    parser.add_argument(
        "--object-name",
        default=None,
        help="Object name for a single file upload. Ignored for folders.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan folders recursively. Enabled by default.",
    )
    parser.add_argument(
        "--init-folders",
        action="store_true",
        help="Create bucket and category folder markers before uploading.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue uploading other files if one file fails.",
    )
    parser.add_argument(
        "--show-categories",
        action="store_true",
        help="Print supported category mapping and exit.",
    )
    args = parser.parse_args()

    if args.show_categories:
        print(json.dumps(RAW_DOCUMENT_CATEGORIES, ensure_ascii=False, indent=2))
        return

    input_path = Path(args.input_path)
    raw_root = Path(args.raw_root)
    files = find_upload_files(input_path, recursive=args.recursive)
    if not files:
        raise SystemExit(f"No supported files found under: {input_path}")

    if args.object_name and len(files) != 1:
        raise SystemExit("--object-name can only be used when uploading one file.")

    if args.init_folders:
        prefixes = ensure_raw_document_prefixes(bucket=args.bucket)
        print(f"Initialized prefixes: {', '.join(prefixes)}")

    results = []
    failures = []
    print(f"Input: {input_path}")
    print(f"Files: {len(files)}")
    print(f"Bucket: {args.bucket or '(from env)'}")
    print(f"Raw root: {raw_root}")
    print(f"Category: {args.category or '(preserve relative path)'}")

    for index, file_path in enumerate(files, start=1):
        print(f"\n[{index}/{len(files)}] {file_path}")
        try:
            result = upload_raw_document_file(
                file_path,
                category=args.category,
                bucket=args.bucket,
                object_name=args.object_name,
                raw_root=raw_root,
            )
        except Exception as exc:
            failures.append((file_path, exc))
            print(f"Failed: {exc}")
            if not args.continue_on_error:
                raise
            continue
        results.append(result)
        print(f"Uploaded: {result.bucket}/{result.object_name}")
        print(f"Size: {result.size}")
        print(f"URL: {result.url}")

    print("\nSummary")
    print(f"Succeeded files: {len(results)}")
    print(f"Failed files: {len(failures)}")
    if failures:
        for file_path, exc in failures:
            print(f"- {file_path}: {exc}")
        raise SystemExit(1)


def find_upload_files(input_path: Path, *, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if is_supported_file(input_path) else []
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Input path must be a file or folder: {input_path}")

    iterator: Iterable[Path] = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(
        path for path in iterator if path.is_file() and is_supported_file(path)
    )


def is_supported_file(path: Path) -> bool:
    return path.is_file() and not path.name.startswith("~$")


if __name__ == "__main__":
    main()
