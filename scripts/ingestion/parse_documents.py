from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.parser.parser import parse_document


SUPPORTED_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".pdf"}


@dataclass
class ParseResult:
    file_path: Path
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
        help="Document file or folder to parse, for example data/raw/普通表格.",
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

    input_path = Path(args.input_path)
    document_files = find_document_files(input_path, recursive=args.recursive)
    if not document_files:
        raise SystemExit(f"No supported documents found under: {input_path}")

    print(f"Input: {input_path}")
    print(f"Documents: {len(document_files)}")
    print("Mode: parse only")

    results: list[ParseResult] = []
    failures: list[tuple[Path, Exception]] = []

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


def find_document_files(input_path: Path, *, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if is_supported_document(input_path) else []

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Input path must be a file or folder: {input_path}")

    iterator: Iterable[Path] = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and is_supported_document(path)
    )


def is_supported_document(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith("~$")


def _processing_txt_file(file_path: Path) -> Path:
    return Path("data") / "processing" / file_path.stem / "txt" / f"{file_path.stem}.txt"


if __name__ == "__main__":
    main()
