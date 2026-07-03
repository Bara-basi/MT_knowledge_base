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

from app.db.neo4j import ensure_neo4j_schema  # noqa: E402
from app.services.graph.graph_store import GraphStoreService  # noqa: E402


DEFAULT_MANIFEST_DIR = Path("data/metadata/graph/standard_product_subgraph")


def load_manifest_metadata(manifest_dir: Path) -> dict[str, object]:
    summary_path = manifest_dir / "summary.json"
    if not summary_path.exists():
        return {}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest summary must be a JSON object: {summary_path}")
    return data


def resolve_sub_graph_name(manifest_dir: Path, override: str | None) -> str | None:
    if override:
        return override
    metadata = load_manifest_metadata(manifest_dir)
    value = metadata.get("sub_graph_name") or metadata.get("subgraph_name")
    return str(value) if value else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Import graph JSONL manifest into Neo4j.")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help="Directory containing nodes.jsonl and edges.jsonl.",
    )
    parser.add_argument("--nodes-file", type=Path, default=None, help="Override nodes JSONL path.")
    parser.add_argument("--edges-file", type=Path, default=None, help="Override edges JSONL path.")
    parser.add_argument("--graph-name", default=None, help="Override logical graph name.")
    parser.add_argument(
        "--sub-graph-name",
        default=None,
        help=(
            "Override sub logical graph name. Defaults to sub_graph_name/subgraph_name "
            "in manifest summary.json."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip constraint creation before import.",
    )
    args = parser.parse_args()

    nodes_file = args.nodes_file or args.manifest_dir / "nodes.jsonl"
    edges_file = args.edges_file or args.manifest_dir / "edges.jsonl"
    if not nodes_file.exists():
        raise FileNotFoundError(f"Nodes file does not exist: {nodes_file}")
    if not edges_file.exists():
        raise FileNotFoundError(f"Edges file does not exist: {edges_file}")

    sub_graph_name = resolve_sub_graph_name(args.manifest_dir, args.sub_graph_name)

    schema_result = None
    if not args.skip_schema:
        schema_result = ensure_neo4j_schema()

    service = GraphStoreService(graph_name=args.graph_name, sub_graph_name=sub_graph_name)
    try:
        result = service.import_manifest(nodes_file, edges_file, batch_size=args.batch_size)
    finally:
        service.close()

    print(
        json.dumps(
            {
                "schema": schema_result,
                "import": result,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
