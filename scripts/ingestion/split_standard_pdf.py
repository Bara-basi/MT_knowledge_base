from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.minio import download_raw_document_to_file, list_raw_document_objects
from app.services.parser.standard_pdf_splitter import split_standard_pdf_document


def main() -> None:
    _configure_utf8_stdio()

    parser = argparse.ArgumentParser(
        description="Split standard PDF documents and upload section PDFs to MinIO.",
    )
    parser.add_argument(
        "input_path",
        help="MinIO PDF object/prefix or local PDF/folder to split.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan folders recursively. Enabled by default.",
    )
    parser.add_argument(
        "--split-prefix",
        help=(
            "Optional MinIO object prefix for split PDFs. Use only when splitting one PDF "
            "or when intentionally placing all outputs under one prefix."
        ),
    )
    parser.add_argument(
        "--title-page-max-chars",
        type=int,
        default=900,
        help="Max text chars for a page to be treated as a standard title page.",
    )
    args = parser.parse_args()

    documents = resolve_pdf_documents(args.input_path, recursive=args.recursive)
    if not documents:
        raise SystemExit(f"No PDF documents found under: {args.input_path}")
    if args.split_prefix and len(documents) > 1:
        print("Warning: --split-prefix is shared by multiple PDFs; later files may overwrite same-named sections.")

    print(f"Input: {args.input_path}")
    print(f"PDF documents: {len(documents)}")

    total_sections = 0
    for index, document in enumerate(documents, start=1):
        print(f"\n[{index}/{len(documents)}] Splitting {document.source_reference}", flush=True)
        sections = split_standard_pdf_document(
            document.local_path,
            source_reference=document.source_reference,
            split_prefix=args.split_prefix,
            title_page_max_chars=args.title_page_max_chars,
        )
        total_sections += len(sections)
        print(f"Sections: {len(sections)}")
        if sections:
            print(f"First source URI: {sections[0].source_uri}")
            print(f"Last source URI: {sections[-1].source_uri}")

    print("\nSummary")
    print(f"PDF documents: {len(documents)}")
    print(f"Sections: {total_sections}")


class SplitInput:
    def __init__(self, local_path: Path, source_reference: str) -> None:
        self.local_path = local_path
        self.source_reference = source_reference


def resolve_pdf_documents(input_path: str, *, recursive: bool) -> list[SplitInput]:
    if _is_minio_source(input_path):
        return resolve_minio_pdf_documents(input_path, recursive=recursive)
    return resolve_local_pdf_documents(Path(input_path), recursive=recursive)


def resolve_minio_pdf_documents(input_path: str, *, recursive: bool) -> list[SplitInput]:
    references = list_raw_document_objects(input_path, recursive=recursive)
    documents: list[SplitInput] = []
    for reference in references:
        if Path(reference.object_name).suffix.lower() != ".pdf":
            continue
        local_path = download_raw_document_to_file(reference.uri)
        documents.append(SplitInput(local_path=local_path, source_reference=reference.uri))
    return sorted(documents, key=lambda item: item.source_reference)


def resolve_local_pdf_documents(input_path: Path, *, recursive: bool) -> list[SplitInput]:
    if input_path.is_file():
        return [SplitInput(input_path, str(input_path))] if input_path.suffix.lower() == ".pdf" else []
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Input path must be a PDF file or folder: {input_path}")

    iterator = input_path.rglob("*.pdf") if recursive else input_path.glob("*.pdf")
    return [
        SplitInput(path, str(path))
        for path in sorted(iterator)
        if path.is_file() and not path.name.startswith("~$")
    ]


def _is_minio_source(value: str) -> bool:
    normalized = value.strip().replace("\\", "/")
    parsed = urlparse(normalized)
    return parsed.scheme in {"minio", "s3"} or not Path(value).exists()


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
