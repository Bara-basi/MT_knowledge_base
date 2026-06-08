from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.parser.parser import parse_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse one Word document and save extracted txt.")
    parser.add_argument("docx_file", help="Path to a .docx file.")
    parser.add_argument(
        "--image-workers",
        type=int,
        default=3,
        help="Max concurrent image description API calls. Default: 3.",
    )
    args = parser.parse_args()

    docx_file = Path(args.docx_file)
    parsed_items = parse_document(
        docx_file,
        image_analysis_workers=args.image_workers,
    )
    txt_path = Path("data") / "processing" / docx_file.stem / "txt" / f"{docx_file.stem}.txt"

    print(f"File: {docx_file}")
    print(f"Parsed items: {len(parsed_items)}")
    print(f"Text output: {txt_path}")


if __name__ == "__main__":
    main()
