import json
import pathlib
import re
import sys
from urllib.parse import urlparse


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.storage.lark_script_credentials import get_lark_credentials  # noqa: E402


BASE_URL = "https://open.feishu.cn/open-apis"
METADATA_DIR = pathlib.Path("./data/metadata/local2lark_mapping")
OUTPUT_NAME = "迈拓思学院（内部）.json"
WIKI_URLS = [
    "https://tmqhw1h9zt.feishu.cn/wiki/IbqqwuJPmifob4kPBHvcls3EnCc",
    "https://tmqhw1h9zt.feishu.cn/wiki/FeCpwdWZYifELFkN8cAcTZWpn6c",
    "https://tmqhw1h9zt.feishu.cn/wiki/HTkAw1scPisx4bkijh3cbsP5nde",
    "https://tmqhw1h9zt.feishu.cn/wiki/TTefwAZgOiQfHakOU4xcs1i5nAp",
    "https://tmqhw1h9zt.feishu.cn/wiki/Mg7Swrva2irY0bk5kZfc52Bxnnh",
    "https://tmqhw1h9zt.feishu.cn/wiki/Abyhwms2Ki7ayTkcQgscwTGenbf",
    "https://tmqhw1h9zt.feishu.cn/wiki/SM1AwtC2ZiIDEAkeXWGcjVbKn6g",
    "https://tmqhw1h9zt.feishu.cn/wiki/YrswwbfcBimcbCkfyQIcfIVHngg",
    "https://tmqhw1h9zt.feishu.cn/wiki/UGyJw1oZpij0iYkRTCuc3SrsnGf"
]
INCLUDE_ROOT = True
SCAN_LINKED_DOCUMENTS = True
SCAN_EMBEDDED_FILES = True
FAIL_ON_DUPLICATE = False
REQUEST_TIMEOUT = 30
FEISHU_DOC_URL_PATTERN = re.compile(
    r"https?://[^\s\"'<>]*(?:feishu\.cn|larksuite\.com)/"
    r"(?:wiki|docx|docs|doc|sheets|base|bitable|slides|file)/"
    r"[A-Za-z0-9]+",
    re.IGNORECASE,
)

class FeishuAPIError(Exception):
    def __init__(self, data):
        super().__init__(data)
        self.data = data
        self.code = data.get("code")
        self.msg = data.get("msg", "")


def get_access_token():
    import httpx

    app_id, app_secret = get_lark_credentials()

    resp = httpx.post(
        f"{BASE_URL}/auth/v3/tenant_access_token/internal",
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=REQUEST_TIMEOUT,
    )
    data = resp.json()

    if data["code"] != 0:
        raise FeishuAPIError(data)

    return data["tenant_access_token"]


def api_get(access_token, path, params=None):
    import httpx

    resp = httpx.get(
        f"{BASE_URL}{path}",
        headers={"Authorization": f"Bearer {access_token}"},
        params=params or {},
        timeout=REQUEST_TIMEOUT,
    )
    data = resp.json()

    if data["code"] != 0:
        raise FeishuAPIError(data)

    return data["data"]


def extract_wiki_token(link):
    parsed = urlparse(link)
    parts = [part for part in parsed.path.split("/") if part]

    if "wiki" not in parts:
        raise ValueError(f"not a wiki link: {link}")

    wiki_index = parts.index("wiki")

    if wiki_index + 1 >= len(parts):
        raise ValueError(f"wiki token is missing: {link}")

    return parts[wiki_index + 1]


def get_node(access_token, token):
    data = api_get(
        access_token,
        "/wiki/v2/spaces/get_node",
        params={"token": token},
    )
    return data["node"]


def get_docx_document(access_token, document_id):
    data = api_get(
        access_token,
        f"/docx/v1/documents/{document_id}",
    )
    return data["document"]


def list_child_nodes(access_token, space_id, parent_node_token):
    page_token = None

    while True:
        params = {
            "parent_node_token": parent_node_token,
            "page_size": 50,
        }

        if page_token:
            params["page_token"] = page_token

        data = api_get(
            access_token,
            f"/wiki/v2/spaces/{space_id}/nodes",
            params=params,
        )

        yield from data.get("items", [])

        if not data.get("has_more"):
            break

        page_token = data.get("page_token")


def list_space_root_nodes(access_token, space_id):
    page_token = None

    while True:
        params = {"page_size": 50}

        if page_token:
            params["page_token"] = page_token

        data = api_get(
            access_token,
            f"/wiki/v2/spaces/{space_id}/nodes",
            params=params,
        )

        yield from data.get("items", [])

        if not data.get("has_more"):
            break

        page_token = data.get("page_token")


def list_document_blocks(access_token, document_id):
    page_token = None

    while True:
        params = {"page_size": 500}

        if page_token:
            params["page_token"] = page_token

        data = api_get(
            access_token,
            f"/docx/v1/documents/{document_id}/blocks",
            params=params,
        )

        yield from data.get("items", [])

        if not data.get("has_more"):
            break

        page_token = data.get("page_token")


def get_sub_page_list_tokens(access_token, node):
    if (node.get("obj_type") or "").lower() != "docx":
        return []

    obj_token = node.get("obj_token")
    if not obj_token:
        return []

    tokens = []

    for block in list_document_blocks(access_token, obj_token):
        sub_page_list = block.get("sub_page_list") or {}
        wiki_token = sub_page_list.get("wiki_token")
        if wiki_token:
            tokens.append(wiki_token)

    return tokens


def extract_feishu_doc_links(value):
    links = []

    def walk(current_value):
        if isinstance(current_value, dict):
            for child_value in current_value.values():
                walk(child_value)
            return

        if isinstance(current_value, list):
            for child_value in current_value:
                walk(child_value)
            return

        if not isinstance(current_value, str):
            return

        for match in FEISHU_DOC_URL_PATTERN.finditer(current_value):
            links.append(clean_extracted_url(match.group(0)))

    walk(value)
    return dedupe_preserve_order(link for link in links if link)


def extract_embedded_files(value):
    files = []

    def walk(current_value):
        if isinstance(current_value, dict):
            file_value = current_value.get("file")
            if isinstance(file_value, dict):
                file_name = str(file_value.get("name") or "").strip()
                file_token = str(file_value.get("token") or "").strip()
                if file_name:
                    files.append({"name": sanitize_file_name(file_name), "token": file_token})

            for child_value in current_value.values():
                walk(child_value)
            return

        if isinstance(current_value, list):
            for child_value in current_value:
                walk(child_value)

    walk(value)
    return dedupe_files(files)


def dedupe_files(files):
    seen = set()
    output = []

    for file_item in files:
        key = (file_item.get("name", ""), file_item.get("token", ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(file_item)

    return output


def clean_extracted_url(url):
    return url.strip().rstrip(".,;:!?，。；：！？、）)]}\"'")


def dedupe_preserve_order(values):
    seen = set()
    output = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)

    return output


def parse_feishu_url(link):
    parsed = urlparse(link)
    parts = [part for part in parsed.path.split("/") if part]

    for index, part in enumerate(parts):
        if part in {"wiki", "docx", "docs", "doc", "sheets", "base", "bitable", "slides", "file"}:
            if index + 1 < len(parts):
                return part, parts[index + 1]

    return "", ""


def get_export_ext(obj_type):
    mapping = {
        "doc": "docx",
        "docx": "docx",
        "sheet": "xlsx",
        "bitable": "xlsx",
        "slides": "pptx",
    }
    return mapping.get((obj_type or "").lower())


def build_file_name(node):
    title = sanitize_file_name(node.get("title", ""))
    obj_type = node.get("obj_type")

    if obj_type == "file":
        return title

    ext = get_export_ext(obj_type)
    if ext is None:
        return title

    suffix = pathlib.Path(title).suffix.lstrip(".")
    if suffix.lower() == ext.lower():
        return title

    return f"{title}.{ext}"


def is_supported_file_node(node):
    obj_type = node.get("obj_type")
    return obj_type == "file" or get_export_ext(obj_type) is not None


def sanitize_file_name(name):
    name = (name or "").strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "", name)
    name = name.strip(" .")
    return name or "untitled"


def sanitize_output_name(name):
    name = sanitize_file_name(name)
    if pathlib.Path(name).suffix.lower() == ".json":
        return name
    return f"{name}.json"


def build_wiki_view_url(node, source_wiki_url):
    url = str(node.get("url") or "").strip()
    if url:
        return url

    parsed = urlparse(source_wiki_url)
    if not parsed.scheme or not parsed.netloc:
        return ""

    node_token = str(node.get("node_token") or "").strip()
    if not node_token:
        return ""

    return f"{parsed.scheme}://{parsed.netloc}/wiki/{node_token}"


def add_mapping_item(mapping, duplicates, file_name, url, node_token=""):
    if not file_name or not url:
        return

    existing_url = mapping.get(file_name)
    if existing_url is None:
        mapping[file_name] = url
        return

    if existing_url != url:
        duplicates.append(
            {
                "file_name": file_name,
                "existing_url": existing_url,
                "duplicate_url": url,
                "node_token": node_token,
            }
        )


def add_node_mapping(mapping, duplicates, node, source_wiki_url):
    if not is_supported_file_node(node):
        return

    add_mapping_item(
        mapping,
        duplicates,
        build_file_name(node),
        build_wiki_view_url(node, source_wiki_url),
        node_token=node.get("node_token", ""),
    )


def add_linked_document_mapping(access_token, mapping, duplicates, link):
    link_type, token = parse_feishu_url(link)
    if not token:
        return False

    if link_type == "wiki":
        node = get_node(access_token, token)
        add_node_mapping(mapping, duplicates, node, link)
        return True

    if link_type == "docx":
        document = get_docx_document(access_token, token)
        title = sanitize_file_name(document.get("title", ""))
        add_mapping_item(
            mapping,
            duplicates,
            build_name_with_ext(title, "docx"),
            link,
            node_token=token,
        )
        return True

    return False


def build_name_with_ext(title, ext):
    file_name = sanitize_file_name(title)
    suffix = pathlib.Path(file_name).suffix.lstrip(".")

    if suffix.lower() == ext.lower():
        return file_name

    return f"{file_name}.{ext}"


def collect_embedded_document_links(access_token, node, mapping, duplicates):
    if not SCAN_LINKED_DOCUMENTS:
        return 0

    if (node.get("obj_type") or "").lower() != "docx":
        return 0

    obj_token = node.get("obj_token")
    if not obj_token:
        return 0

    links = []

    for block in list_document_blocks(access_token, obj_token):
        links.extend(extract_feishu_doc_links(block))

    links = dedupe_preserve_order(links)
    if not links:
        return 0

    success_count = 0
    print(f"Found {len(links)} embedded Feishu document links in: {node.get('title', '')}")

    for link in links:
        try:
            if add_linked_document_mapping(access_token, mapping, duplicates, link):
                success_count += 1
        except Exception as exc:
            print(f"Skipped embedded link: {link} -> {exc}")

    return success_count


def collect_embedded_files(access_token, node, source_wiki_url, mapping, duplicates):
    if not SCAN_EMBEDDED_FILES:
        return 0

    if (node.get("obj_type") or "").lower() != "docx":
        return 0

    obj_token = node.get("obj_token")
    if not obj_token:
        return 0

    files = []

    for block in list_document_blocks(access_token, obj_token):
        files.extend(extract_embedded_files(block))

    files = dedupe_files(files)
    if not files:
        return 0

    page_url = build_wiki_view_url(node, source_wiki_url)
    print(f"Found {len(files)} embedded files in: {node.get('title', '')}")

    for file_item in files:
        add_mapping_item(
            mapping,
            duplicates,
            file_item["name"],
            page_url,
            node_token=file_item.get("token", ""),
        )

    return len(files)


def collect_filename_links(access_token, node, source_wiki_url, include_self=False):
    mapping = {}
    duplicates = []

    def add_node(current_node):
        add_node_mapping(mapping, duplicates, current_node, source_wiki_url)

    def walk(current_node, should_add):
        if should_add:
            add_node(current_node)

        if not current_node.get("has_child"):
            collect_embedded_document_links(
                access_token,
                current_node,
                mapping,
                duplicates,
            )
            collect_embedded_files(
                access_token,
                current_node,
                source_wiki_url,
                mapping,
                duplicates,
            )
            return

        for child in list_child_nodes(
            access_token,
            current_node["space_id"],
            current_node["node_token"],
        ):
            print(f"Found: {build_file_name(child)}")
            walk(child, True)

    walk(node, include_self)
    return mapping, duplicates


def collect_filename_links_from_roots(access_token, nodes, source_wiki_url):
    mapping = {}
    duplicates = []

    for node in nodes:
        node_mapping, node_duplicates = collect_filename_links(
            access_token,
            node,
            source_wiki_url,
            include_self=True,
        )
        for file_name, url in node_mapping.items():
            existing_url = mapping.get(file_name)
            if existing_url is None:
                mapping[file_name] = url
            elif existing_url != url:
                duplicates.append(
                    {
                        "file_name": file_name,
                        "existing_url": existing_url,
                        "duplicate_url": url,
                        "node_token": node.get("node_token", ""),
                    }
                )
        duplicates.extend(node_duplicates)

    return mapping, duplicates


def read_metadata(output_path):
    if not output_path.exists():
        return {}

    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"metadata file must contain a JSON object: {output_path}")

    return {str(key): str(value) for key, value in data.items()}


def merge_metadata(existing_mapping, new_mapping):
    merged = dict(existing_mapping)
    duplicates = []
    added_count = 0

    for file_name, url in new_mapping.items():
        existing_url = merged.get(file_name)
        if existing_url is None:
            merged[file_name] = url
            added_count += 1
            continue

        if existing_url != url:
            duplicates.append(
                {
                    "file_name": file_name,
                    "existing_url": existing_url,
                    "duplicate_url": url,
                    "node_token": "",
                }
            )

    return merged, duplicates, added_count


def write_metadata(mapping, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_duplicates(duplicates, output_path, append=True):
    if not duplicates:
        return None

    duplicate_path = output_path.with_name(f"{output_path.stem}.duplicates.json")
    if append and duplicate_path.exists():
        with duplicate_path.open("r", encoding="utf-8") as f:
            existing_duplicates = json.load(f)
        if not isinstance(existing_duplicates, list):
            existing_duplicates = []
        duplicates = existing_duplicates + duplicates

    duplicate_path.write_text(
        json.dumps(duplicates, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return duplicate_path


def collect_from_wiki_url(access_token, wiki_url):
    root_token = extract_wiki_token(wiki_url)
    root_node = get_node(access_token, root_token)
    print(f"Root node: {root_node.get('title', '')}")

    sub_page_list_tokens = []

    if not root_node.get("has_child") and not INCLUDE_ROOT:
        sub_page_list_tokens = get_sub_page_list_tokens(access_token, root_node)

    if sub_page_list_tokens:
        print(
            "Root node is a docx page with a sub-page-list block. "
            "Trying to list wiki space root nodes instead of saving the root page."
        )
        print(f"Sub-page-list tokens found: {', '.join(sub_page_list_tokens)}")

        try:
            root_nodes = list(list_space_root_nodes(access_token, root_node["space_id"]))
        except FeishuAPIError as exc:
            mapping = {}
            duplicates = []
            print(
                "Cannot list wiki space root nodes through OpenAPI. "
                f"code={exc.code}, msg={exc.msg}"
            )
            print(
                "This wiki homepage cannot be expanded with the current app token. "
                "Use a child folder/page link as the start URL, or grant the app "
                "read permission to the whole wiki space."
            )
        else:
            root_nodes = [
                node
                for node in root_nodes
                if node.get("node_token") != root_node.get("node_token")
            ]
            mapping, duplicates = collect_filename_links_from_roots(
                access_token,
                root_nodes,
                wiki_url,
            )
    else:
        include_self = INCLUDE_ROOT or not root_node.get("has_child")

        if include_self and not INCLUDE_ROOT:
            print("Root node has no children; including the root document itself.")

        mapping, duplicates = collect_filename_links(
            access_token,
            root_node,
            wiki_url,
            include_self=include_self,
        )

    return mapping, duplicates


def main():
    output_path = METADATA_DIR / sanitize_output_name(OUTPUT_NAME)
    print(f"Output: {output_path}")

    if not WIKI_URLS:
        print("No wiki links configured. Fill WIKI_URLS in this script first.")
        return 1

    access_token = get_access_token()
    merged_mapping = read_metadata(output_path)
    all_duplicates = []
    total_added = 0
    failed = []

    print(f"Loaded {len(merged_mapping)} existing filename links.")

    for index, wiki_url in enumerate(WIKI_URLS, start=1):
        print(f"[{index}/{len(WIKI_URLS)}] Processing: {wiki_url}")

        try:
            mapping, duplicates = collect_from_wiki_url(access_token, wiki_url)
        except Exception as exc:
            failed.append({"wiki_url": wiki_url, "reason": str(exc)})
            print(f"Failed: {wiki_url} -> {exc}")
            continue

        merged_mapping, merge_duplicates, added_count = merge_metadata(
            merged_mapping,
            mapping,
        )
        all_duplicates.extend(duplicates)
        all_duplicates.extend(merge_duplicates)
        total_added += added_count
        write_metadata(merged_mapping, output_path)

        print(
            f"Saved progress: discovered {len(mapping)}, "
            f"added {added_count}, total {len(merged_mapping)}."
        )

    if all_duplicates and FAIL_ON_DUPLICATE:
        duplicate_path = write_duplicates(all_duplicates, output_path)
        print(f"Duplicate file names found. Details: {duplicate_path}")
        return 1

    duplicate_path = write_duplicates(all_duplicates, output_path)

    print(f"Added {total_added} filename links to {output_path}")
    if duplicate_path:
        print(
            "Duplicate file names were kept as the first discovered link. "
            f"Details: {duplicate_path}"
        )
    if failed:
        failed_path = output_path.with_name(f"{output_path.stem}.failed.json")
        failed_path.write_text(
            json.dumps(failed, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Failed links: {len(failed)}. Details: {failed_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
