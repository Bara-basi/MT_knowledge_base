from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_file(PROJECT_ROOT / ".env")

from app.db.neo4j import (  # noqa: E402
    check_neo4j_health,
    delete_graph,
    ensure_neo4j_schema,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize Neo4j for MTSCO knowledge graph.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check Neo4j connectivity; do not create constraints.",
    )
    parser.add_argument(
        "--delete-graph",
        action="store_true",
        help="Delete all nodes and relationships for the configured logical graph.",
    )
    parser.add_argument(
        "--graph-name",
        default=None,
        help="Override the logical graph name when deleting. Defaults to NEO4J_GRAPH_NAME.",
    )
    args = parser.parse_args()

    if args.check_only:
        result = check_neo4j_health()
    elif args.delete_graph:
        result = delete_graph(args.graph_name)
    else:
        result = ensure_neo4j_schema()

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
