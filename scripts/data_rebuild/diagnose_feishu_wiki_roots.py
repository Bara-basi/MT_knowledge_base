"""Read-only diagnostic for Feishu Wiki root-page traversal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.lark_client import (  # noqa: E402
    get_access_token,
    get_node,
    get_sub_page_list_tokens,
    list_child_nodes,
    list_space_root_nodes,
    node_title,
    parse_feishu_url,
)


def diagnose(access_token: str, url: str) -> dict[str, object]:
    link_type, token = parse_feishu_url(url)
    if link_type != "wiki" or not token:
        raise ValueError(f"not a Feishu Wiki URL: {url}")
    node = get_node(access_token, token)
    children = list(list_child_nodes(access_token, node["space_id"], node["node_token"]))
    sub_page_tokens = get_sub_page_list_tokens(access_token, node)
    space_roots = list(list_space_root_nodes(access_token, node["space_id"]))
    return {
        "url": url,
        "title": node_title(node),
        "has_child": bool(node.get("has_child")),
        "direct_child_count": len(children),
        "sub_page_list_count": len(sub_page_tokens),
        "space_root_node_count": len(space_roots),
        "requires_space_root_expansion": bool(sub_page_tokens) and not bool(node.get("has_child")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", action="append", required=True, help="Feishu Wiki URL; repeat for multiple URLs.")
    args = parser.parse_args()
    token = get_access_token()
    print(json.dumps([diagnose(token, url) for url in args.url], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
