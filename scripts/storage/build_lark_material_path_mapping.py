"""Build a Feishu Wiki marketing-material catalogue in PostgreSQL.

This script reads the ``wiki`` mapping in a source JSON file, walks every
configured Wiki node recursively, and upserts the paths and links into the
marketing-material catalogue. It does not export or download document content.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Iterator
from urllib.parse import urlparse

import requests


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.storage.lark_script_credentials import get_lark_credentials  # noqa: E402
from app.services.marketing_asset_catalog import (  # noqa: E402
    TABLE_NAME,
    ensure_marketing_asset_catalog,
    prepare_asset_row,
    upsert_marketing_assets,
)


BASE_URL = "https://open.feishu.cn/open-apis"
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "src" / "material.json"
REQUEST_TIMEOUT = 30


class FeishuAPIError(RuntimeError):
    """An unsuccessful response from the Feishu Open API."""

    def __init__(self, data: dict) -> None:
        super().__init__(f"Feishu API error {data.get('code')}: {data.get('msg', data)}")
        self.data = data


def get_access_token() -> str:
    app_id, app_secret = get_lark_credentials()
    response = requests.post(
        f"{BASE_URL}/auth/v3/tenant_access_token/internal",
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=REQUEST_TIMEOUT,
    )
    data = response.json()
    if data.get("code") != 0:
        raise FeishuAPIError(data)
    return data["tenant_access_token"]


def api_get(access_token: str, path: str, params: dict | None = None) -> dict:
    response = requests.get(
        f"{BASE_URL}{path}",
        headers={"Authorization": f"Bearer {access_token}"},
        params=params or {},
        timeout=REQUEST_TIMEOUT,
    )
    data = response.json()
    if data.get("code") != 0:
        raise FeishuAPIError(data)
    return data["data"]


def extract_wiki_token(link: str) -> str:
    parts = [part for part in urlparse(link).path.split("/") if part]
    try:
        return parts[parts.index("wiki") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"not a valid Feishu wiki link: {link}") from exc


def get_node(access_token: str, token: str) -> dict:
    return api_get(access_token, "/wiki/v2/spaces/get_node", {"token": token})["node"]


def list_child_nodes(access_token: str, space_id: str, parent_node_token: str) -> Iterator[dict]:
    page_token = None
    while True:
        params = {"parent_node_token": parent_node_token, "page_size": 50}
        if page_token:
            params["page_token"] = page_token
        data = api_get(access_token, f"/wiki/v2/spaces/{space_id}/nodes", params)
        yield from data.get("items", [])
        if not data.get("has_more"):
            return
        page_token = data.get("page_token")


def node_title(node: dict) -> str:
    return str(node.get("title") or "untitled").strip() or "untitled"


def build_wiki_view_url(node: dict, source_url: str) -> str:
    """Return the node's own view URL, falling back to a constructed Wiki URL."""
    if url := str(node.get("url") or "").strip():
        return url
    parsed = urlparse(source_url)
    token = str(node.get("node_token") or "").strip()
    if parsed.scheme and parsed.netloc and token:
        return f"{parsed.scheme}://{parsed.netloc}/wiki/{token}"
    return source_url


def walk_nodes(
    access_token: str,
    node: dict,
    source_url: str,
    parent_path: tuple[str, ...],
    visited: set[str],
) -> Iterator[dict[str, str]]:
    """Yield the current node and all descendants with their Wiki paths."""
    node_token = str(node.get("node_token") or "")
    if node_token and node_token in visited:
        return
    if node_token:
        visited.add(node_token)

    path_parts = (*parent_path, node_title(node))
    yield {"path": "/".join(path_parts), "feishu_link": build_wiki_view_url(node, source_url)}

    if not node.get("has_child"):
        return
    space_id = str(node.get("space_id") or "")
    if not space_id or not node_token:
        raise ValueError(f"node cannot list children: {node}")
    for child in list_child_nodes(access_token, space_id, node_token):
        yield from walk_nodes(access_token, child, source_url, path_parts, visited)


def load_wiki_sources(source_path: pathlib.Path) -> dict[str, dict[str, str]]:
    with source_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    wiki = data.get("wiki")
    if not isinstance(wiki, dict):
        raise ValueError(f"{source_path} must contain a 'wiki' object")
    result: dict[str, dict[str, str]] = {}
    for library_name, links in wiki.items():
        if not isinstance(links, dict):
            raise ValueError(f"wiki.{library_name} must be an object of name-to-link values")
        result[str(library_name)] = {str(name): str(link) for name, link in links.items()}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Feishu marketing-material PostgreSQL catalogue.")
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_SOURCE, help="Source JSON containing a wiki mapping.")
    parser.add_argument("--table-name", default=TABLE_NAME, help=f"Target PostgreSQL table. Default: {TABLE_NAME}.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = load_wiki_sources(args.source)
    access_token = get_access_token()
    ensure_marketing_asset_catalog(args.table_name)
    failures: list[str] = []
    total_count = 0

    for library_name, roots in sources.items():
        rows: list[dict[str, str]] = []
        for configured_name, link in roots.items():
            try:
                root_node = get_node(access_token, extract_wiki_token(link))
                for record in walk_nodes(access_token, root_node, link, (), set()):
                    rows.append(
                        prepare_asset_row(
                            library_name=library_name,
                            path=record["path"],
                            feishu_link=record["feishu_link"],
                            source_file=str(args.source),
                        )
                    )
            except Exception as exc:  # Continue so accessible roots are still catalogued.
                failures.append(f"{library_name}/{configured_name}: {exc}")
                print(f"FAILED {library_name}/{configured_name}: {exc}", file=sys.stderr)
        count = upsert_marketing_assets(rows, table_name=args.table_name)
        total_count += count
        print(f"Upserted {count} records for {library_name} into {args.table_name}")

    if failures:
        print(f"Completed with {len(failures)} failed root(s).", file=sys.stderr)
        return 1
    print(f"Completed: upserted {total_count} marketing-material records into {args.table_name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
