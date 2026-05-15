from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.vector_store import VectorStoreService


def main() -> None:
    parser = argparse.ArgumentParser(description="Upsert embedded chunks into Milvus.")
    parser.add_argument("document_name", help="Name under data/processing/[document_name].")
    parser.add_argument(
        "--no-flush",
        action="store_true",
        help="Skip flushing after upsert. Useful for larger batch jobs.",
    )
    args = parser.parse_args()

    embedding_file = (
        Path("data")
        / "processing"
        / args.document_name
        / "embedding"
        / f"{args.document_name}.embeddings.json"
    )
    result = VectorStoreService().upsert_embedding_file(
        embedding_file,
        flush=not args.no_flush,
    )

    print(f"Collection: {result['collection_name']}")
    print(f"Input: {result['input_file']}")
    print(f"Upsert count: {result['upsert_count']}")
    if result["ids"]:
        print(f"First id: {result['ids'][0]}")
        print(f"Last id: {result['ids'][-1]}")


if __name__ == "__main__":
    main()
