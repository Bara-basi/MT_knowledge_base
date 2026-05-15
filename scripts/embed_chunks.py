from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.embedding import EmbeddingService


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed saved chunk JSON files.")
    parser.add_argument("document_name", help="Name under data/processing/[document_name].")
    args = parser.parse_args()

    output_path = EmbeddingService().embed_processing_chunks(args.document_name)
    print(f"Embedding output: {output_path}")


if __name__ == "__main__":
    main()
