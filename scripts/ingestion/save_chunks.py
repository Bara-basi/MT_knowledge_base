from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.chunking.splitter import save_processing_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Save parsed txt chunks as JSON.")
    parser.add_argument("document_name", help="Name under data/processing/[document_name].")
    args = parser.parse_args()

    output_path = save_processing_chunks(args.document_name)
    print(f"Chunk output: {output_path}")


if __name__ == "__main__":
    main()
