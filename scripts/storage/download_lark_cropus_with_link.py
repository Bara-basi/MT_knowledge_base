import argparse
import os
import pathlib
import re
import sys
import time
from urllib.parse import urlparse

import requests


BASE_URL = "https://open.feishu.cn/open-apis"
DEFAULT_WIKI_URL = "https://tmqhw1h9zt.feishu.cn/wiki/K3SnwkJ5eiI1pAkyM4IcaQqCnSh"
DOWNLOAD_DIR = pathlib.Path("./data/raw")

REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 120
MAX_EXPORT_WAIT_SECONDS = 180
EXPORT_FAILED_GRACE_SECONDS = 30
MAX_FILE_ATTEMPTS = 3

# Keep the same fallback credentials as download_lark_cropus_in_table.py.
# Environment variables take precedence so this script can be reused safely.
DEFAULT_APP_ID = "cli_aa895b208df9dcdd"
DEFAULT_APP_SECRET = "AfESVeY5n9m7By2plKh97g05C7TtbCAZ"


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
        raise Exception(data)

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
        raise Exception(data)

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
        raise Exception(data)

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
            raise Exception(data)

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
    resp = requests.get(
        f"{BASE_URL}/drive/v1/export_tasks/file/{file_token}/download",
        headers={"Authorization": f"Bearer {access_token}"},
        stream=True,
        timeout=DOWNLOAD_TIMEOUT,
    )
    resp.raise_for_status()
    write_stream(resp, save_path)


def download_drive_file(access_token, file_token, save_path):
    resp = requests.get(
        f"{BASE_URL}/drive/v1/files/{file_token}/download",
        headers={"Authorization": f"Bearer {access_token}"},
        stream=True,
        timeout=DOWNLOAD_TIMEOUT,
    )
    resp.raise_for_status()
    write_stream(resp, save_path)


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


def resolve_save_path(folder, node, ext, used_paths):
    save_path = folder / build_save_name(node.get("title", ""), ext)
    key = str(save_path.resolve()).lower()

    if key not in used_paths:
        used_paths.add(key)
        return save_path

    stem = save_path.stem
    suffix = save_path.suffix
    node_token = node.get("node_token", "")[:8] or "duplicate"
    deduped_path = save_path.with_name(f"{stem}_{node_token}{suffix}")
    used_paths.add(str(deduped_path.resolve()).lower())
    return deduped_path


def build_child_folder(folder, title):
    child_folder = folder / sanitize_path_part(title)

    if child_folder.exists() and child_folder.is_file():
        child_folder = folder / f"{sanitize_path_part(title)}_children"

    return child_folder


def download_node(access_token, node, folder, used_paths, overwrite=False):
    obj_type = node.get("obj_type")
    obj_token = node.get("obj_token")

    if not obj_token:
        return False, "missing obj_token"

    if obj_type == "file":
        ext = pathlib.Path(node.get("title", "")).suffix.lstrip(".") or "bin"
        save_path = resolve_save_path(folder, node, ext, used_paths)

        if save_path.exists() and save_path.stat().st_size > 0 and not overwrite:
            return True, f"已存在: {save_path}"

        download_drive_file(access_token, obj_token, save_path)
        return True, str(save_path)

    export_type = get_export_type(obj_type)
    ext = get_export_ext(obj_type)

    if export_type is None or ext is None:
        return False, f"unsupported obj_type={obj_type}"

    save_path = resolve_save_path(folder, node, ext, used_paths)

    if save_path.exists() and save_path.stat().st_size > 0 and not overwrite:
        return True, f"已存在: {save_path}"

    ticket = create_export_task(access_token, obj_token, export_type, ext)
    file_token = wait_export(access_token, ticket, obj_token)
    download_exported_file(access_token, file_token, save_path)
    return True, str(save_path)


def walk_and_download(access_token, node, folder, used_paths, include_self=True, overwrite=False):
    success_count = 0
    skipped = []
    failed = []
    title = node.get("title", "")

    if include_self:
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
                    success_count += 1
                    print(f"完成: {result}")
                    last_error = None
                    break

                last_error = result
                break

            except Exception as exc:
                last_error = str(exc)
                print(
                    f"失败: {title}，第 {attempt}/{MAX_FILE_ATTEMPTS} 次尝试 -> {exc}"
                )

                if attempt < MAX_FILE_ATTEMPTS:
                    time.sleep(2 * attempt)

        if last_error:
            item = {
                "title": title,
                "path": str(folder),
                "url": node.get("url", ""),
                "obj_type": node.get("obj_type", ""),
                "reason": last_error,
            }

            if last_error.startswith("unsupported"):
                skipped.append(item)
                print(f"跳过: {title} ({last_error})")
            else:
                failed.append(item)

    if not node.get("has_child"):
        return success_count, skipped, failed

    child_folder = build_child_folder(folder, title)
    child_folder.mkdir(parents=True, exist_ok=True)

    for child in list_child_nodes(
        access_token,
        node["space_id"],
        node["node_token"],
    ):
        print(f"处理节点: {child_folder / sanitize_path_part(child.get('title', ''))}")
        child_success, child_skipped, child_failed = walk_and_download(
            access_token,
            child,
            child_folder,
            used_paths,
            include_self=True,
            overwrite=overwrite,
        )
        success_count += child_success
        skipped.extend(child_skipped)
        failed.extend(child_failed)

    return success_count, skipped, failed


def print_problem_summary(title, items):
    if not items:
        return

    print(title)

    for index, item in enumerate(items, start=1):
        print(f"{index}. {item['title']}")
        print(f"   路径: {item['path']}")
        print(f"   链接: {item['url']}")
        print(f"   类型: {item['obj_type']}")
        print(f"   原因: {item['reason']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Feishu/Lark wiki documents recursively and preserve paths."
    )
    parser.add_argument(
        "wiki_url",
        nargs="?",
        default=DEFAULT_WIKI_URL,
        help="Feishu wiki link, defaults to the current corpus link.",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DOWNLOAD_DIR),
        help="Directory where downloaded documents are saved.",
    )
    parser.add_argument(
        "--include-root",
        action="store_true",
        help="Also export the document represented by the provided link itself.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even when the target path already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_token = extract_wiki_token(args.wiki_url)
    download_root = pathlib.Path(args.download_dir)

    access_token = get_access_token()
    root_node = get_node(access_token, root_token)
    root_folder = download_root / sanitize_path_part(root_node.get("title", "wiki"))

    print(f"根节点: {root_node.get('title', '')}")
    print(f"保存目录: {root_folder}")

    success_count, skipped, failed = walk_and_download(
        access_token,
        root_node,
        root_folder.parent,
        used_paths=set(),
        include_self=args.include_root,
        overwrite=args.overwrite,
    )

    print(
        f"下载完成，成功 {success_count} 个，跳过 {len(skipped)} 个，失败 {len(failed)} 个。"
    )
    print_problem_summary("跳过清单:", skipped)
    print_problem_summary("失败清单:", failed)

    if failed:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
