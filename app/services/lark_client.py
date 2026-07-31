from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from app.core.config import settings


BASE_URL = "https://open.feishu.cn/open-apis"
REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 120
MAX_EXPORT_WAIT_SECONDS = 180
EXPORT_FAILED_GRACE_SECONDS = 30


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


def get_access_token() -> str:
    app_id = settings.feishu_app_id
    app_secret = settings.feishu_app_secret
    if not app_id or not app_secret:
        raise RuntimeError(
            "Missing required Lark credentials in .env: "
            "LARK_APP_ID, LARK_APP_SECRET"
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


def api_get(access_token: str, path: str, params=None):
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


def api_post(access_token: str, path: str, json_body: dict, params=None):
    resp = requests.post(
        f"{BASE_URL}{path}",
        headers={"Authorization": f"Bearer {access_token}"},
        params=params or {},
        json=json_body,
        timeout=REQUEST_TIMEOUT,
    )
    data = resp.json()
    if data["code"] != 0:
        raise FeishuAPIError(data)
    return data["data"]


def parse_feishu_url(link: str) -> tuple[str, str]:
    parsed = urlparse(link)
    parts = [part for part in parsed.path.split("/") if part]
    for index, part in enumerate(parts):
        if part in {"wiki", "docx", "docs", "doc", "sheets", "base", "bitable", "slides", "file"}:
            if index + 1 < len(parts):
                return part, parts[index + 1]
    return "", ""


def get_node(access_token: str, token: str):
    data = api_get(access_token, "/wiki/v2/spaces/get_node", params={"token": token})
    return data["node"]


def get_docx_document(access_token: str, document_id: str):
    data = api_get(access_token, f"/docx/v1/documents/{document_id}")
    return data["document"]


def list_child_nodes(access_token: str, space_id: str, parent_node_token: str):
    page_token = None
    while True:
        params = {"parent_node_token": parent_node_token, "page_size": 50}
        if page_token:
            params["page_token"] = page_token
        data = api_get(access_token, f"/wiki/v2/spaces/{space_id}/nodes", params=params)
        yield from data.get("items", [])
        if not data.get("has_more"):
            break
        page_token = data.get("page_token")


def list_space_root_nodes(access_token: str, space_id: str):
    page_token = None
    while True:
        params = {"page_size": 50}
        if page_token:
            params["page_token"] = page_token
        data = api_get(access_token, f"/wiki/v2/spaces/{space_id}/nodes", params=params)
        yield from data.get("items", [])
        if not data.get("has_more"):
            break
        page_token = data.get("page_token")


def list_document_blocks(access_token: str, document_id: str):
    page_token = None
    while True:
        params = {"page_size": 500}
        if page_token:
            params["page_token"] = page_token
        data = api_get(access_token, f"/docx/v1/documents/{document_id}/blocks", params=params)
        yield from data.get("items", [])
        if not data.get("has_more"):
            break
        page_token = data.get("page_token")


def get_sub_page_list_tokens(access_token: str, node: dict) -> list[str]:
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


def collect_wiki_root_nodes(access_token: str, root_node: dict):
    if root_node.get("has_child"):
        return None
    sub_page_list_tokens = get_sub_page_list_tokens(access_token, root_node)
    if not sub_page_list_tokens:
        return None
    root_nodes = list(list_space_root_nodes(access_token, root_node["space_id"]))
    return [
        node
        for node in root_nodes
        if node.get("node_token") != root_node.get("node_token")
    ]


def get_export_ext(obj_type: str | None) -> str | None:
    return {
        "doc": "docx",
        "docx": "docx",
        "sheet": "xlsx",
        "bitable": "xlsx",
        "slides": "pptx",
    }.get((obj_type or "").lower())


def get_export_type(obj_type: str | None) -> str | None:
    return {
        "doc": "doc",
        "docx": "docx",
        "sheet": "sheet",
        "bitable": "bitable",
        "slides": "slides",
    }.get((obj_type or "").lower())


def create_export_task(access_token: str, token: str, export_type: str | None, ext: str | None):
    if export_type is None or ext is None:
        return None
    resp = requests.post(
        f"{BASE_URL}/drive/v1/export_tasks",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"token": token, "type": export_type, "file_extension": ext},
        timeout=REQUEST_TIMEOUT,
    )
    data = resp.json()
    if data["code"] != 0:
        raise FeishuAPIError(data)
    return data["data"]["ticket"]


def wait_export(access_token: str, ticket: str, token: str):
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
            raise RuntimeError(f"export task failed: ticket={ticket}, token={token}, result={result}")
        if job_status == 2 and elapsed > EXPORT_FAILED_GRACE_SECONDS:
            raise RuntimeError(
                f"export task stayed failed after {EXPORT_FAILED_GRACE_SECONDS}s: "
                f"ticket={ticket}, token={token}, result={result}"
            )
        if elapsed > MAX_EXPORT_WAIT_SECONDS:
            raise TimeoutError(
                f"export task timeout after {MAX_EXPORT_WAIT_SECONDS}s: "
                f"ticket={ticket}, token={token}, last_result={last_result}"
            )
        time.sleep(2)


def download_exported_file(access_token: str, file_token: str, save_path: Path) -> None:
    url = f"{BASE_URL}/drive/v1/export_tasks/file/{file_token}/download"
    resp = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, stream=True, timeout=DOWNLOAD_TIMEOUT)
    raise_for_download_error(resp, url)
    write_stream(resp, save_path)


def download_drive_file(access_token: str, file_token: str, save_path: Path) -> None:
    url = f"{BASE_URL}/drive/v1/files/{file_token}/download"
    resp = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, stream=True, timeout=DOWNLOAD_TIMEOUT)
    raise_for_download_error(resp, url)
    write_stream(resp, save_path)


def raise_for_download_error(resp, url: str) -> None:
    if resp.status_code < 400:
        return
    try:
        text = resp.text
    except Exception:
        text = "<response body unavailable>"
    raise FeishuHTTPError(resp.status_code, url, text)


def write_stream(resp, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as handle:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def sanitize_path_part(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" .")
    return name or "untitled"


def build_save_name(title: str, ext: str) -> str:
    file_name = sanitize_path_part(title)
    suffix = Path(file_name).suffix.lstrip(".")
    if suffix.lower() == ext.lower():
        return file_name
    return f"{file_name}.{ext}"


def resolve_save_path(folder: Path, title: str, ext: str, dedupe_key: str, used_paths: set[str]) -> Path:
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


def node_title(node: dict, fallback: str = "untitled") -> str:
    return str(node.get("title") or fallback).strip() or fallback


def build_wiki_view_url(node: dict, source_url: str) -> str:
    url = str(node.get("url") or "").strip()
    if url:
        return url
    parsed = urlparse(source_url)
    node_token = str(node.get("node_token") or "").strip()
    if not parsed.scheme or not parsed.netloc or not node_token:
        return source_url
    return f"{parsed.scheme}://{parsed.netloc}/wiki/{node_token}"


def download_node(
    access_token: str,
    node: dict,
    folder: Path,
    used_paths: set[str],
    overwrite: bool = False,
) -> tuple[bool, str]:
    obj_type = node.get("obj_type")
    obj_token = node.get("obj_token")
    title = node_title(node)
    if not obj_token:
        return False, "missing obj_token"

    if obj_type == "file":
        ext = Path(title).suffix.lstrip(".") or "bin"
        save_path = resolve_save_path(folder, title, ext, node.get("node_token") or obj_token, used_paths)
        if save_path.exists() and save_path.stat().st_size > 0 and not overwrite:
            return True, f"exists: {save_path}"
        download_drive_file(access_token, obj_token, save_path)
        return True, str(save_path)

    export_type = get_export_type(obj_type)
    ext = get_export_ext(obj_type)
    if export_type is None or ext is None:
        return False, f"unsupported obj_type={obj_type}"

    save_path = resolve_save_path(folder, title, ext, node.get("node_token") or obj_token, used_paths)
    if save_path.exists() and save_path.stat().st_size > 0 and not overwrite:
        return True, f"exists: {save_path}"

    ticket = create_export_task(access_token, obj_token, export_type, ext)
    file_token = wait_export(access_token, ticket, obj_token)
    download_exported_file(access_token, file_token, save_path)
    return True, str(save_path)


def load_vector_sources(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"vector source file must contain a JSON object: {path}")
    return {
        "single_file": normalize_source_mapping(data.get("single_file", {}), "single_file"),
        "wiki": normalize_source_mapping(data.get("wiki", {}), "wiki"),
    }


def normalize_source_mapping(value, source_type: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{source_type} must be a JSON object")
    return {str(name): str(link) for name, link in value.items()}


def create_bitable_record(
    access_token: str,
    *,
    app_token: str,
    table_id: str,
    fields: dict,
    field_key: str | None = None,
    user_id_type: str | None = None,
) -> dict:
    params = {}
    if field_key:
        params["field_key"] = field_key
    if user_id_type:
        params["user_id_type"] = user_id_type
    return api_post(
        access_token,
        f"/bitable/v1/apps/{app_token}/tables/{table_id}/records",
        {"fields": fields},
        params=params,
    ).get("record", {})
