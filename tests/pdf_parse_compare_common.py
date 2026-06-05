from __future__ import annotations

import json
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_CASES = [
    PROJECT_ROOT
    / "data"
    / "raw"
    / "structed_pdf"
    / "嘉兴迈拓不锈钢有限公司 规章制度.pdf",
    PROJECT_ROOT
    / "data"
    / "raw"
    / "unstructed_pdf"
    / "2024 FPSO P80 P83 DUPLEX STEEL SUPPLY.pdf",
]

OUTPUT_ROOT = PROJECT_ROOT / "data" / "processing" / "pdf_parser_compare"


def safe_name(value: str) -> str:
    value = re.sub(r'[\\/:*?"<>|]+', "_", value.strip())
    value = re.sub(r"\s+", " ", value)
    return value or "unnamed"


def text_stats(text: str) -> dict[str, int]:
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "chars": len(text),
        "non_empty_lines": len(lines),
    }


def build_page(page_number: int, text: str, **extra: Any) -> dict[str, Any]:
    page = {
        "page_number": page_number,
        "text": text or "",
        **text_stats(text or ""),
    }
    page.update(extra)
    return page


def save_result(parser_name: str, pdf_path: Path, pages: list[dict[str, Any]]) -> Path:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    full_text = "\n\n".join(page.get("text", "") for page in pages)
    output_dir = OUTPUT_ROOT / parser_name / safe_name(pdf_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / f"{safe_name(pdf_path.stem)}.{parser_name}.txt"
    json_path = output_dir / f"{safe_name(pdf_path.stem)}.{parser_name}.json"

    txt_path.write_text(full_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "parser": parser_name,
                "source_pdf": str(pdf_path.relative_to(PROJECT_ROOT)),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "summary": {
                    "pages": len(pages),
                    **text_stats(full_text),
                },
                "pages": pages,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[{parser_name}] {pdf_path.name}: "
        f"{len(pages)} pages, {len(full_text)} chars -> {output_dir}"
    )
    return output_dir


def save_error_result(parser_name: str, pdf_path: Path, error: Exception) -> Path:
    output_dir = OUTPUT_ROOT / parser_name / safe_name(pdf_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / f"{safe_name(pdf_path.stem)}.{parser_name}.txt"
    json_path = output_dir / f"{safe_name(pdf_path.stem)}.{parser_name}.json"

    message = f"{type(error).__name__}: {error}"
    txt_path.write_text(f"PARSE_FAILED\n{message}\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "parser": parser_name,
                "source_pdf": str(pdf_path.relative_to(PROJECT_ROOT)),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "summary": {
                    "pages": 0,
                    "chars": 0,
                    "non_empty_lines": 0,
                    "status": "failed",
                },
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "pages": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[{parser_name}] {pdf_path.name}: PARSE_FAILED -> {output_dir}")
    return output_dir


def run_parser(parser_name: str, parse_pdf) -> None:
    print(f"Output root: {OUTPUT_ROOT}")
    for pdf_path in PDF_CASES:
        try:
            pages = parse_pdf(pdf_path)
        except Exception as error:
            save_error_result(parser_name, pdf_path, error)
            continue
        save_result(parser_name, pdf_path, pages)
