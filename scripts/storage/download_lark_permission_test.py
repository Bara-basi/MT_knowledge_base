import argparse
import json
import os
import pathlib
import re
import sys
import time
from urllib.parse import urlparse

import requests


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE_URL = "https://open.feishu.cn/open-apis"
DEFAULT_VECTOR_SRC = pathlib.Path("./data/src/vector_src.json")
DOWNLOAD_DIR = pathlib.Path("./data/temp")
NO_PERMISSION_FILE = "no_permission.json"
FAILED_FILE = "download_failed.json"

REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 120
MAX_EXPORT_WAIT_SECONDS = 180
EXPORT_FAILED_GRACE_SECONDS = 30
MAX_FILE_ATTEMPTS = 3

# Keep the same fallback credentials as download_lark_cropus_with_link.py.
# Environment variables take precedence so this script can be reused safely.
DEFAULT_APP_ID = "cli_aa895b208df9dcdd"
DEFAULT_APP_SECRET = "AfESVeY5n9m7By2plKh97g05C7TtbCAZ"


def load_env_file(path):
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_file(PROJECT_ROOT / ".env")


class FeishuAPIError(Exception):
    def __init__(self, data):
        super().__init__(data)
        self.data = data
        self.code = data.get("code")
        self.msg = data.get("msg", "")


class FeishuHTTPError(Exception):
    def __init__(self, status_code, url, text):
        message = f"HTTP {status_code}: {url}: {text[:300]}"
        super().__init__(message)
        self.status_code = status_code
        self.url = url
        self.text = text


def get_access_token():
    app_id = os.getenv("FEISHU_APP_ID") or os.getenv("LARK_APP_ID") or DEFAULT_APP_ID
    app_secret = (
        os.getenv("FEISHU_APP_SECRET")
        or os.getenv("LARK_APP_SECRET")
        or DEFAULT_APP_SECRET
    )

    resp = requests.post(
        f"{BASE_URL}/auth/v3/tenant_access_token/internal",
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=REQUEST_TIMEOUT,
    )
    data = resp.json()

    if data["code"] != 0:
        raise FeishuAPIError(data)

    return data["tenant_access_token"]


def api_get(access_token, path, params=None):
    resp = requests.get(
        f"{BASE_URL}{path}",
        headers={"Authorization": f"Bearer {access_token}"},
        params=params or {},
        timeout=REQUEST_TIMEOUT,
    )
    data = resp.json()

    if data["code"] != 0:
        raise FeishuAPIError(data)

    return data["data"]


def parse_feishu_url(link):
    parsed = urlparse(link)
    parts = [part for part in parsed.path.split("/") if part]

    for index, part in enumerate(parts):
        if part in {"wiki", "docx", "docs", "doc", "sheets", "base", "bitable", "slides", "file"}:
            if index + 1 < len(parts):
                return part, parts[index + 1]

    return "", ""


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


def get_export_ext(obj_type):
    mapping = {
        "doc": "docx",
        "docx": "docx",
        "sheet": "xlsx",
        "bitable": "xlsx",
        "slides": "pptx",
    }
    return mapping.get((obj_type or "").lower())


def get_export_type(obj_type):
    mapping = {
        "doc": "doc",
        "docx": "docx",
        "sheet": "sheet",
        "bitable": "bitable",
        "slides": "slides",
    }
    return mapping.get((obj_type or "").lower())


def create_export_task(access_token, token, export_type, ext):
    if export_type is None or ext is None:
        return None

    resp = requests.post(
        f"{BASE_URL}/drive/v1/export_tasks",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "token": token,
            "type": export_type,
            "file_extension": ext,
        },
        timeout=REQUEST_TIMEOUT,
    )
    data = resp.json()

    if data["code"] != 0:
        raise FeishuAPIError(data)

    return data["data"]["ticket"]


def wait_export(access_token, ticket, token):
    started_at = time.monotonic()
    last_result = None

    while True:
        resp = requests.get(
            f"{BASE_URL}/drive/v1/export_tasks/{ticket}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"token": token},
            timeout=REQUEST_TIMEOUT,
        )
        data = resp.json()

        if data["code"] != 0:
            raise FeishuAPIError(data)

        result = data["data"]["result"]
        last_result = result
        job_status = result["job_status"]

        if job_status == 0:
            return result["file_token"]

        elapsed = time.monotonic() - started_at

        if job_status == 2 and result.get("job_error_msg"):
            raise Exception(
                f"export task failed: ticket={ticket}, token={token}, result={result}"
            )

        if job_status == 2 and elapsed > EXPORT_FAILED_GRACE_SECONDS:
            raise Exception(
                "export task stayed failed after "
                f"{EXPORT_FAILED_GRACE_SECONDS}s: "
                f"ticket={ticket}, token={token}, result={result}"
            )

        if elapsed > MAX_EXPORT_WAIT_SECONDS:
            raise TimeoutError(
                "export task timeout after "
                f"{MAX_EXPORT_WAIT_SECONDS}s: "
                f"ticket={ticket}, token={token}, last_result={last_result}"
            )

        time.sleep(2)


def download_exported_file(access_token, file_token, save_path):
    url = f"{BASE_URL}/drive/v1/export_tasks/file/{file_token}/download"
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        stream=True,
        timeout=DOWNLOAD_TIMEOUT,
    )
    raise_for_download_error(resp, url)
    write_stream(resp, save_path)


def download_drive_file(access_token, file_token, save_path):
    url = f"{BASE_URL}/drive/v1/files/{file_token}/download"
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        stream=True,
        timeout=DOWNLOAD_TIMEOUT,
    )
    raise_for_download_error(resp, url)
    write_stream(resp, save_path)


def raise_for_download_error(resp, url):
    if resp.status_code < 400:
        return

    text = ""
    try:
        text = resp.text
    except Exception:
        text = "<response body unavailable>"
    raise FeishuHTTPError(resp.status_code, url, text)


def write_stream(resp, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def sanitize_path_part(name):
    name = (name or "").strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" .")
    return name or "untitled"


def build_save_name(title, ext):
    file_name = sanitize_path_part(title)
    suffix = pathlib.Path(file_name).suffix.lstrip(".")

    if suffix.lower() == ext.lower():
        return file_name

    return f"{file_name}.{ext}"


def resolve_save_path(folder, title, ext, dedupe_key, used_paths):
    save_path = folder / build_save_name(title, ext)
    key = str(save_path.resolve()).lower()

    if key not in used_paths:
        used_paths.add(key)
        return save_path

    stem = save_path.stem
    suffix = save_path.suffix
    dedupe_suffix = (dedupe_key or "duplicate")[:8]
    deduped_path = save_path.with_name(f"{stem}_{dedupe_suffix}{suffix}")
    used_paths.add(str(deduped_path.resolve()).lower())
    return deduped_path


def build_child_folder(folder, title):
    child_folder = folder / sanitize_path_part(title)

    if child_folder.exists() and child_folder.is_file():
        child_folder = folder / f"{sanitize_path_part(title)}_children"

    return child_folder


def node_title(node, fallback="untitled"):
    return str(node.get("title") or fallback).strip() or fallback


def build_wiki_view_url(node, source_url):
    url = str(node.get("url") or "").strip()
    if url:
        return url

    parsed = urlparse(source_url)
    node_token = str(node.get("node_token") or "").strip()
    if not parsed.scheme or not parsed.netloc or not node_token:
        return source_url

    return f"{parsed.scheme}://{parsed.netloc}/wiki/{node_token}"


def download_node(access_token, node, folder, used_paths, overwrite=False):
    obj_type = node.get("obj_type")
    obj_token = node.get("obj_token")
    title = node_title(node)

    if not obj_token:
        return False, "missing obj_token"

    if obj_type == "file":
        ext = pathlib.Path(title).suffix.lstrip(".") or "bin"
        save_path = resolve_save_path(
            folder,
            title,
            ext,
            node.get("node_token") or obj_token,
            used_paths,
        )

        if save_path.exists() and save_path.stat().st_size > 0 and not overwrite:
            return True, f"exists: {save_path}"

        download_drive_file(access_token, obj_token, save_path)
        return True, str(save_path)

    export_type = get_export_type(obj_type)
    ext = get_export_ext(obj_type)

    if export_type is None or ext is None:
        return False, f"unsupported obj_type={obj_type}"

    save_path = resolve_save_path(
        folder,
        title,
        ext,
        node.get("node_token") or obj_token,
        used_paths,
    )

    if save_path.exists() and save_path.stat().st_size > 0 and not overwrite:
        return True, f"exists: {save_path}"

    ticket = create_export_task(access_token, obj_token, export_type, ext)
    file_token = wait_export(access_token, ticket, obj_token)
    download_exported_file(access_token, file_token, save_path)
    return True, str(save_path)


def download_direct_link(access_token, source_name, link, folder, used_paths, overwrite=False):
    link_type, token = parse_feishu_url(link)
    if not token:
        raise ValueError(f"unsupported Feishu link: {link}")

    if link_type == "wiki":
        node = get_node(access_token, token)
        if not node.get("url"):
            node["url"] = link
        return download_node_with_retries(
            access_token,
            node,
            folder,
            used_paths,
            source_type="single_file",
            source_name=source_name,
            source_url=link,
            path_titles=[source_name],
            overwrite=overwrite,
        )

    if link_type == "docx":
        title = source_name
        try:
            document = get_docx_document(access_token, token)
            title = document.get("title") or title
        except Exception:
            pass
        node = {
            "title": title,
            "obj_type": "docx",
            "obj_token": token,
            "node_token": token,
            "url": link,
        }
        return download_node_with_retries(
            access_token,
            node,
            folder,
            used_paths,
            source_type="single_file",
            source_name=source_name,
            source_url=link,
            path_titles=[source_name],
            overwrite=overwrite,
        )

    obj_type_by_link = {
        "docs": "doc",
        "doc": "doc",
        "sheets": "sheet",
        "base": "bitable",
        "bitable": "bitable",
        "slides": "slides",
        "file": "file",
    }
    obj_type = obj_type_by_link.get(link_type)
    if not obj_type:
        raise ValueError(f"unsupported Feishu link type={link_type}: {link}")

    node = {
        "title": source_name,
        "obj_type": obj_type,
        "obj_token": token,
        "node_token": token,
        "url": link,
    }
    return download_node_with_retries(
        access_token,
        node,
        folder,
        used_paths,
        source_type="single_file",
        source_name=source_name,
        source_url=link,
        path_titles=[source_name],
        overwrite=overwrite,
    )


def download_node_with_retries(
    access_token,
    node,
    folder,
    used_paths,
    *,
    source_type,
    source_name,
    source_url,
    path_titles,
    overwrite=False,
):
    title = node_title(node)
    last_error = None

    for attempt in range(1, MAX_FILE_ATTEMPTS + 1):
        try:
            processed, result = download_node(
                access_token,
                node,
                folder,
                used_paths,
                overwrite=overwrite,
            )

            if processed:
                print(f"Downloaded: {result}")
                return 1, [], []

            last_error = Exception(result)
            break
        except Exception as exc:
            last_error = exc
            print(f"Failed: {title}, attempt {attempt}/{MAX_FILE_ATTEMPTS} -> {exc}")

            if attempt < MAX_FILE_ATTEMPTS:
                time.sleep(2 * attempt)

    item = build_problem_item(
        last_error,
        source_type=source_type,
        source_name=source_name,
        source_url=source_url,
        node=node,
        path_titles=path_titles,
    )
    if is_permission_error(last_error):
        return 0, [item], []
    return 0, [], [item]


def walk_leaf_nodes_and_download(
    access_token,
    node,
    folder,
    used_paths,
    *,
    source_type,
    source_name,
    source_url,
    path_titles=None,
    overwrite=False,
):
    path_titles = list(path_titles or [])
    title = node_title(node)
    current_titles = path_titles + [title]

    if not node.get("has_child"):
        return download_node_with_retries(
            access_token,
            node,
            folder,
            used_paths,
            source_type=source_type,
            source_name=source_name,
            source_url=source_url,
            path_titles=current_titles,
            overwrite=overwrite,
        )

    child_folder = build_child_folder(folder, title)
    success_count = 0
    no_permission = []
    failed = []

    try:
        children = list(
            list_child_nodes(
                access_token,
                node["space_id"],
                node["node_token"],
            )
        )
    except Exception as exc:
        item = build_problem_item(
            exc,
            source_type=source_type,
            source_name=source_name,
            source_url=source_url,
            node=node,
            path_titles=current_titles,
            stage="list_child_nodes",
        )
        if is_permission_error(exc):
            return 0, [item], []
        return 0, [], [item]

    if not children:
        return download_node_with_retries(
            access_token,
            node,
            folder,
            used_paths,
            source_type=source_type,
            source_name=source_name,
            source_url=source_url,
            path_titles=current_titles,
            overwrite=overwrite,
        )

    child_folder.mkdir(parents=True, exist_ok=True)

    for child in children:
        print(f"Visiting leaf path: {child_folder / sanitize_path_part(node_title(child))}")
        child_success, child_no_permission, child_failed = walk_leaf_nodes_and_download(
            access_token,
            child,
            child_folder,
            used_paths,
            source_type=source_type,
            source_name=source_name,
            source_url=source_url,
            path_titles=current_titles,
            overwrite=overwrite,
        )
        success_count += child_success
        no_permission.extend(child_no_permission)
        failed.extend(child_failed)

    return success_count, no_permission, failed


def collect_wiki_root_nodes(access_token, root_node):
    if root_node.get("has_child"):
        return None

    try:
        sub_page_list_tokens = get_sub_page_list_tokens(access_token, root_node)
    except Exception:
        raise

    if not sub_page_list_tokens:
        return None

    root_nodes = list(list_space_root_nodes(access_token, root_node["space_id"]))
    return [
        node
        for node in root_nodes
        if node.get("node_token") != root_node.get("node_token")
    ]


def download_wiki_source(access_token, source_name, link, download_root, used_paths, overwrite=False):
    link_type, token = parse_feishu_url(link)
    if link_type != "wiki" or not token:
        raise ValueError(f"wiki source must be a wiki link: {link}")

    root_node = get_node(access_token, token)
    root_node.setdefault("url", link)
    source_folder = download_root / sanitize_path_part(source_name)
    success_count = 0
    no_permission = []
    failed = []

    try:
        root_nodes = collect_wiki_root_nodes(access_token, root_node)
    except Exception as exc:
        item = build_problem_item(
            exc,
            source_type="wiki",
            source_name=source_name,
            source_url=link,
            node=root_node,
            path_titles=[source_name, node_title(root_node)],
            stage="detect_space_root",
        )
        if is_permission_error(exc):
            return 0, [item], []
        return 0, [], [item]

    if root_nodes is not None:
        print(
            "Detected wiki space root page; listing root nodes through "
            f"/wiki/v2/spaces/{root_node['space_id']}/nodes"
        )
        for child in root_nodes:
            child_success, child_no_permission, child_failed = walk_leaf_nodes_and_download(
                access_token,
                child,
                source_folder,
                used_paths,
                source_type="wiki",
                source_name=source_name,
                source_url=link,
                path_titles=[source_name],
                overwrite=overwrite,
            )
            success_count += child_success
            no_permission.extend(child_no_permission)
            failed.extend(child_failed)
        return success_count, no_permission, failed

    return walk_leaf_nodes_and_download(
        access_token,
        root_node,
        download_root,
        used_paths,
        source_type="wiki",
        source_name=source_name,
        source_url=link,
        path_titles=[],
        overwrite=overwrite,
    )


def load_vector_sources(path):
    with pathlib.Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"vector source file must contain a JSON object: {path}")

    return {
        "single_file": normalize_source_mapping(data.get("single_file", {}), "single_file"),
        "wiki": normalize_source_mapping(data.get("wiki", {}), "wiki"),
    }


def normalize_source_mapping(value, source_type):
    if value is None:
        return {}

    if not isinstance(value, dict):
        raise ValueError(f"{source_type} must be a JSON object")

    return {str(name): str(link) for name, link in value.items()}


def build_problem_item(
    exc,
    *,
    source_type,
    source_name,
    source_url,
    node=None,
    path_titles=None,
    stage="download",
):
    node = node or {}
    reason = str(exc)
    item = {
        "source_type": source_type,
        "source_name": source_name,
        "source_url": source_url,
        "title": node_title(node, source_name),
        "path": path_titles or [],
        "url": node.get("url") or build_wiki_view_url(node, source_url),
        "obj_type": node.get("obj_type", ""),
        "obj_token": node.get("obj_token", ""),
        "node_token": node.get("node_token", ""),
        "space_id": node.get("space_id", ""),
        "stage": stage,
        "reason": reason,
    }

    if isinstance(exc, FeishuAPIError):
        item["code"] = exc.code
        item["msg"] = exc.msg
    elif isinstance(exc, FeishuHTTPError):
        item["status_code"] = exc.status_code

    return item


def is_permission_error(exc):
    if isinstance(exc, FeishuHTTPError):
        return exc.status_code in {401, 403}

    if isinstance(exc, FeishuAPIError):
        text = f"{exc.code} {exc.msg} {exc.data}".lower()
    else:
        text = str(exc).lower()

    permission_markers = [
        "permission",
        "forbidden",
        "unauthorized",
        "access denied",
        "no access",
        "no permission",
        "scope",
        "auth",
        "权限",
        "无权",
        "无权限",
        "未授权",
        "没有权限",
        "暂无权限",
    ]
    if any(marker in text for marker in permission_markers):
        return True

    if isinstance(exc, FeishuAPIError):
        return str(exc.code) in {
            "99991661",
            "99991663",
            "99991664",
            "99991665",
            "99991672",
        }

    return False


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_problem_summary(title, items):
    if not items:
        return

    print(title)
    for index, item in enumerate(items, start=1):
        print(f"{index}. {item.get('title', '')}")
        print(f"   source: {item.get('source_name', '')}")
        print(f"   url: {item.get('url', '')}")
        print(f"   stage: {item.get('stage', '')}")
        print(f"   reason: {item.get('reason', '')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download single Feishu files and leaf wiki documents from vector_src.json, "
            "recording inaccessible documents."
        )
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_VECTOR_SRC),
        help="Path to vector_src.json.",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DOWNLOAD_DIR),
        help="Directory where downloaded test files and no_permission.json are saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even when the target path already exists.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return a non-zero exit code when non-permission failures are found.",
    )
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Only check Feishu wiki access paths; do not download files.",
    )
    return parser.parse_args()


def diagnose_wiki_access(access_token, sources):
    has_failure = False

    for index, (source_name, link) in enumerate(sources["wiki"].items(), start=1):
        print(f"[diagnose {index}/{len(sources['wiki'])}] {source_name}")
        link_type, token = parse_feishu_url(link)
        if link_type != "wiki" or not token:
            print(f"  invalid wiki link: {link}")
            has_failure = True
            continue

        try:
            node = get_node(access_token, token)
        except Exception as exc:
            print(f"  get_node: failed -> {exc}")
            has_failure = True
            continue

        print(
            "  get_node: ok "
            f"title={node_title(node)} "
            f"space_id={node.get('space_id')} "
            f"has_child={node.get('has_child')} "
            f"obj_type={node.get('obj_type')}"
        )

        try:
            children = list(
                list_child_nodes(
                    access_token,
                    node["space_id"],
                    node["node_token"],
                )
            )
        except Exception as exc:
            print(f"  list_child_nodes(parent_node_token): failed -> {exc}")
            has_failure = True
        else:
            print(f"  list_child_nodes(parent_node_token): ok count={len(children)}")

        sub_page_tokens = []
        if (node.get("obj_type") or "").lower() == "docx" and node.get("obj_token"):
            try:
                sub_page_tokens = get_sub_page_list_tokens(access_token, node)
            except Exception as exc:
                print(f"  read sub_page_list blocks: failed -> {exc}")
                has_failure = True
            else:
                print(f"  read sub_page_list blocks: ok count={len(sub_page_tokens)}")

        try:
            root_nodes = list(list_space_root_nodes(access_token, node["space_id"]))
        except Exception as exc:
            print(f"  list_space_root_nodes(space): failed -> {exc}")
            has_failure = True
        else:
            root_titles = [node_title(root_node) for root_node in root_nodes[:5]]
            print(
                "  list_space_root_nodes(space): ok "
                f"count={len(root_nodes)} first={root_titles}"
            )

        if sub_page_tokens:
            print(
                "  note: this root document contains a sub-page-list block; "
                "expanding it requires list_space_root_nodes(space)."
            )

    return 1 if has_failure else 0


def main():
    args = parse_args()
    download_root = pathlib.Path(args.download_dir)
    sources = load_vector_sources(args.source)
    access_token = get_access_token()
    used_paths = set()
    success_count = 0
    no_permission = []
    failed = []

    print(f"Source: {args.source}")
    print(f"Download dir: {download_root}")

    if args.diagnose_only:
        return diagnose_wiki_access(access_token, sources)

    for index, (source_name, link) in enumerate(sources["single_file"].items(), start=1):
        print(f"[single_file {index}/{len(sources['single_file'])}] {source_name}")
        try:
            count, source_no_permission, source_failed = download_direct_link(
                access_token,
                source_name,
                link,
                download_root / "single_file",
                used_paths,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            item = build_problem_item(
                exc,
                source_type="single_file",
                source_name=source_name,
                source_url=link,
                path_titles=[source_name],
                stage="resolve_source",
            )
            count = 0
            source_no_permission = [item] if is_permission_error(exc) else []
            source_failed = [] if is_permission_error(exc) else [item]

        success_count += count
        no_permission.extend(source_no_permission)
        failed.extend(source_failed)
        write_json(download_root / NO_PERMISSION_FILE, no_permission)

    for index, (source_name, link) in enumerate(sources["wiki"].items(), start=1):
        print(f"[wiki {index}/{len(sources['wiki'])}] {source_name}")
        try:
            count, source_no_permission, source_failed = download_wiki_source(
                access_token,
                source_name,
                link,
                download_root,
                used_paths,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            item = build_problem_item(
                exc,
                source_type="wiki",
                source_name=source_name,
                source_url=link,
                path_titles=[source_name],
                stage="resolve_source",
            )
            count = 0
            source_no_permission = [item] if is_permission_error(exc) else []
            source_failed = [] if is_permission_error(exc) else [item]

        success_count += count
        no_permission.extend(source_no_permission)
        failed.extend(source_failed)
        write_json(download_root / NO_PERMISSION_FILE, no_permission)

    no_permission_path = download_root / NO_PERMISSION_FILE
    failed_path = download_root / FAILED_FILE
    write_json(no_permission_path, no_permission)
    write_json(failed_path, failed)

    print(
        f"Done. downloaded={success_count}, "
        f"no_permission={len(no_permission)}, failed={len(failed)}"
    )
    print(f"No permission report: {no_permission_path}")
    if failed:
        print(f"Other failures: {failed_path}")

    print_problem_summary("No permission:", no_permission)
    print_problem_summary("Other failures:", failed)

    if failed and args.fail_on_error:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
