from __future__ import annotations

from pathlib import Path

try:
    from app.services.parser.word_parser import parse_word_document
except ModuleNotFoundError:
    from word_parser import parse_word_document


def parse_document(
    file_path: str | Path,
    *,
    image_analysis_workers: int = 3,
):
    """Route an input document to the matching parser by file extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".docx":
        return parse_word_document(path, image_analysis_workers=image_analysis_workers)

    raise ValueError(f"Unsupported document type: {suffix or 'unknown'}")


if __name__ == "__main__":
    target_file = Path("data") / "raw" / "process_guide" / "订阅号运营SOP.docx"
    parsed_items = parse_document(target_file)
    txt_path = Path("data") / "processing" / target_file.stem / "txt" / f"{target_file.stem}.txt"

    print(f"File: {target_file}")
    print(f"Parsed text blocks: {len(parsed_items)}")
    print(f"Text output: {txt_path}")
