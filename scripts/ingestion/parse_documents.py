from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.parser.parser import is_generated_standard_pdf_section, parse_document
from app.services.parser.paths import processing_subdir
from app.db.minio import list_raw_document_objects, parse_raw_document_reference


SUPPORTED_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".pdf"}


@dataclass
class ParseResult:
    file_path: str
    txt_file: Path
    item_count: int


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_utf8_stdio()

    parser = argparse.ArgumentParser(
        description="Parse supported documents only, using app.services.parser.parser.parse_document.",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="",
        help="MinIO document object or prefix to parse. Default: bucket root.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan folders recursively. Enabled by default.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue parsing other documents when one document fails.",
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=3,
        help="Max concurrent image description API calls per document. Default: 3.",
    )
    args = parser.parse_args()

    input_path = args.input_path
    document_files = find_document_files(input_path, recursive=args.recursive)
    if not document_files:
        raise SystemExit(f"No supported documents found under: {input_path}")

    print(f"Input: {input_path}")
    print(f"Documents: {len(document_files)}")
    print("Mode: parse only")

    results: list[ParseResult] = []
    failures: list[tuple[str, Exception]] = []

    for index, file_path in enumerate(document_files, start=1):
        print(f"\n[{index}/{len(document_files)}] Parsing {file_path}", flush=True)
        try:
            parsed_items = parse_document(
                file_path,
                image_analysis_workers=args.image_workers,
            )
        except Exception as exc:
            failures.append((file_path, exc))
            print(f"Failed: {exc}", flush=True)
            if not args.continue_on_error:
                raise
            continue

        txt_file = _processing_txt_file(file_path)
        result = ParseResult(
            file_path=file_path,
            txt_file=txt_file,
            item_count=len(parsed_items),
        )
        results.append(result)
        print(f"Items: {result.item_count}")
        print(f"Text output: {result.txt_file}")

    print("\nSummary")
    print(f"Succeeded documents: {len(results)}")
    print(f"Failed documents: {len(failures)}")
    print(f"Total parsed items: {sum(result.item_count for result in results)}")

    if failures:
        print("\nFailures")
        for file_path, exc in failures:
            print(f"- {file_path}: {exc}")
        raise SystemExit(1)


def find_document_files(input_path: str | Path, *, recursive: bool) -> list[str]:
    objects = list_raw_document_objects(str(input_path or ""), recursive=recursive)
    return sorted(
        reference.uri
        for reference in objects
        if is_supported_document(reference.object_name)
        and not is_generated_standard_pdf_section(reference.object_name)
    )


def is_supported_document(path: str | Path) -> bool:
    source_path = _source_parts(path)
    return source_path.suffix.lower() in SUPPORTED_EXTENSIONS and not source_path.name.startswith("~$")


def _processing_txt_file(file_path: str | Path) -> Path:
    source_path = _source_parts(file_path)
    return processing_subdir(file_path, "txt") / f"{source_path.stem}.txt"


def _source_parts(path: str | Path) -> PurePosixPath:
    value = str(path).replace("\\", "/")
    if value.startswith("minio://") or value.startswith("s3://") or "data/raw/" in value.lower():
        return PurePosixPath(parse_raw_document_reference(value).object_name)
    return PurePosixPath(value)


if __name__ == "__main__":
    main()
