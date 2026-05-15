from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.retrieval import RetrievalService


def main() -> None:
    parser = argparse.ArgumentParser(description="Test vector recall from Milvus.")
    parser.add_argument(
        "query",
        nargs="?",
        default="如何制作订阅号推文？",
        help="User query to embed and search.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of hits to return.")
    parser.add_argument(
        "--bm25-model",
        default=str(
            Path("data")
            / "processing"
            / "订阅号运营SOP"
            / "embedding"
            / "订阅号运营SOP.bm25.json"
        ),
        help="BM25 model JSON saved during embedding generation.",
    )
    args = parser.parse_args()

    results = RetrievalService().search(
        args.query,
        limit=args.limit,
        bm25_model_file=args.bm25_model,
    )

    print(f"Query: {args.query}")
    print(f"BM25 model: {args.bm25_model}")
    print(f"Results: {len(results)}")
    for index, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank: {index}")
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
