import json
import re
import time
import pathlib
import requests

BASE_URL = "https://open.feishu.cn/open-apis"
REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 120
MAX_EXPORT_WAIT_SECONDS = 180
EXPORT_FAILED_GRACE_SECONDS = 30
MAX_FILE_ATTEMPTS = 3


# =========================APP_ID = "cli_xxx"
APP_TOKEN = "B8Ceb24adaJ2XIsPWxucpgNonhh"
TABLE_ID = "tblDAQhe2iVHFKUu"

DOWNLOAD_DIR = r"./data/raw"
# 获取 tenant_access_token
# =========================

def get_access_token():
    resp = requests.post(
        f"{BASE_URL}/auth/v3/tenant_access_token/internal",
        json={
            "app_id": "cli_aa895b208df9dcdd",
            "app_secret": "AfESVeY5n9m7By2plKh97g05C7TtbCAZ"
        },
        timeout=REQUEST_TIMEOUT
    )

    data = resp.json()

    if data["code"] != 0:
        raise Exception(data)

    return data["tenant_access_token"]


# =========================
# 获取全部记录
# =========================

def get_records(access_token):

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    page_token = None

    records = []

    while True:

        params = {
            "page_size": 500
        }

        if page_token:
            params["page_token"] = page_token

        resp = requests.post(
            f"{BASE_URL}/bitable/v1/apps/{APP_TOKEN}/tables/{TABLE_ID}/records/search",
            headers=headers,
            params=params,
            json={},
            timeout=REQUEST_TIMEOUT
        )

        data = resp.json()

        if data["code"] != 0:
            raise Exception(data)

        items = data["data"]["items"]

        records.extend(items)

        if not data["data"]["has_more"]:
            break

        page_token = data["data"]["page_token"]

    return records


# =========================
# 飞书类型映射
# =========================

def get_export_ext(mention_type):

    mapping = {
        "Doc": "docx",
        "Docx": "docx",
        "Sheet": "xlsx",
        "Bitable": "xlsx",
        "Slides": "pptx",
        "doc": "docx",
        "docx": "docx",
        "sheet": "xlsx",
        "bitable": "xlsx",
        "slides": "pptx"
    }

    return mapping.get(mention_type)


def get_export_type(mention_type):

    mapping = {
        "Doc": "doc",
        "Docx": "docx",
        "Sheet": "sheet",
        "Bitable": "bitable",
        "Slides": "slides",
        "doc": "doc",
        "docx": "docx",
        "sheet": "sheet",
        "bitable": "bitable",
        "slides": "slides"
    }

    return mapping.get(mention_type)


def resolve_file_ref(access_token, item):

    mention_type = item.get("mentionType")
    token = item.get("token")
    real_mention_type = item.get("realMentionType")

    if mention_type != "Wiki":
        return token, get_export_type(mention_type), get_export_ext(mention_type)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=utf-8"
    }

    resp = requests.get(
        f"{BASE_URL}/wiki/v2/spaces/get_node",
        headers=headers,
        params={
            "token": token
        },
        timeout=REQUEST_TIMEOUT
    )

    data = resp.json()

    if data["code"] != 0:
        raise Exception(data)

    node = data["data"]["node"]
    obj_token = node.get("obj_token")
    obj_type = node.get("obj_type") or get_export_type(real_mention_type)

    if obj_type == "file" or real_mention_type == "Box":
        return obj_token, "file", pathlib.Path(item.get("text", "")).suffix.lstrip(".")

    return obj_token, obj_type, get_export_ext(obj_type)


# =========================
# 创建导出任务
# =========================

def create_export_task(
        access_token,
        token,
        export_type,
        ext
):

    if export_type is None or ext is None:
        return None

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    payload = {
        "token": token,
        "type": export_type,
        "file_extension": ext
    }

    resp = requests.post(
        f"{BASE_URL}/drive/v1/export_tasks",
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )

    data = resp.json()

    if data["code"] != 0:
        raise Exception(data)

    return data["data"]["ticket"]


# =========================
# 查询导出任务
# =========================

def wait_export(
        access_token,
        ticket,
        token
):

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    started_at = time.monotonic()
    last_result = None

    while True:

        resp = requests.get(
            f"{BASE_URL}/drive/v1/export_tasks/{ticket}",
            headers=headers,
            params={
                "token": token
            },
            timeout=REQUEST_TIMEOUT
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


# =========================
# 下载导出文件
# =========================

def download_file(
        access_token,
        file_token,
        save_path
):

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    resp = requests.get(
        f"{BASE_URL}/drive/v1/export_tasks/file/{file_token}/download",
        headers=headers,
        stream=True,
        timeout=DOWNLOAD_TIMEOUT
    )

    resp.raise_for_status()

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(
                chunk_size=1024 * 1024
        ):
            f.write(chunk)


def download_drive_file(
        access_token,
        file_token,
        save_path
):

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    resp = requests.get(
        f"{BASE_URL}/drive/v1/files/{file_token}/download",
        headers=headers,
        stream=True,
        timeout=DOWNLOAD_TIMEOUT
    )

    resp.raise_for_status()

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(
                chunk_size=1024 * 1024
        ):
            f.write(chunk)


def sanitize_filename(name):
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name)
    return name or "untitled"


def build_save_name(title, ext):
    file_name = sanitize_filename(title)
    suffix = pathlib.Path(file_name).suffix.lstrip(".")
    if suffix.lower() == ext.lower():
        return file_name
    return f"{file_name}.{ext}"


def build_summary_item(doc_type, item, reason):
    return {
        "doc_type": doc_type,
        "title": item.get("text", ""),
        "link": item.get("link", ""),
        "mention_type": item.get("mentionType", ""),
        "real_mention_type": item.get("realMentionType", ""),
        "reason": reason,
    }


def print_problem_summary(title, items):
    if not items:
        return

    print(title)
    for index, item in enumerate(items, start=1):
        print(f"{index}. [{item['doc_type']}] {item['title']}")
        print(f"   链接: {item['link']}")
        print(
            "   类型: "
            f"mentionType={item['mention_type']}, "
            f"realMentionType={item['real_mention_type']}"
        )
        print(f"   原因: {item['reason']}")


def process_file(access_token, download_root, doc_type, item):
    title = item["text"]

    token, export_type, ext = resolve_file_ref(
        access_token,
        item
    )

    if token is None or export_type is None or ext is None:
        return False, "unsupported file type"

    print(
        f"开始处理: {title}"
    )

    folder = (
        download_root /
        doc_type
    )

    save_path = (
        folder /
        build_save_name(title, ext)
    )

    if export_type == "file":
        download_drive_file(
            access_token,
            token,
            save_path
        )

    else:
        ticket = create_export_task(
            access_token,
            token,
            export_type,
            ext
        )

        file_token = wait_export(
            access_token,
            ticket,
            token
        )

        download_file(
            access_token,
            file_token,
            save_path
        )

    print(
        f"完成: {save_path}"
    )

    return True, None


# =========================
# 主流程
# =========================

def main():

    download_root = pathlib.Path(
        DOWNLOAD_DIR
    )

    access_token = get_access_token()

    records = get_records(
        access_token
    )

    success_count = 0
    skipped = []
    failed = []

    print(
        f"发现 {len(records)} 条记录"
    )

    for record in records:

        fields = record["fields"]

        doc_type = ""

        if fields.get("文档类型"):
            doc_type = fields["文档类型"][0]

        related_files = fields.get(
            "相关文件",
            []
        )

        for item in related_files:

            title = item["text"]

            last_error = None

            for attempt in range(1, MAX_FILE_ATTEMPTS + 1):
                try:
                    processed, reason = process_file(
                        access_token,
                        download_root,
                        doc_type,
                        item
                    )

                    if not processed:
                        last_error = reason
                        break

                    success_count += 1
                    last_error = None
                    break

                except Exception as exc:
                    last_error = str(exc)
                    print(
                        f"失败: {title}，第 {attempt}/{MAX_FILE_ATTEMPTS} 次尝试 -> {exc}"
                    )

                    if attempt < MAX_FILE_ATTEMPTS:
                        time.sleep(2 * attempt)

            if last_error == "unsupported file type":
                skipped.append(
                    build_summary_item(
                        doc_type,
                        item,
                        last_error
                    )
                )
                print(
                    f"跳过: {title}"
                )
                continue

            if last_error is not None:
                failed.append(
                    build_summary_item(
                        doc_type,
                        item,
                        last_error
                    )
                )
                continue

    print(
        f"下载完成，成功 {success_count} 个，跳过 {len(skipped)} 个，失败 {len(failed)} 个"
    )

    if skipped:
        print_problem_summary(
            "跳过清单（需要手动确认类型）:",
            skipped
        )

    if failed:
        print_problem_summary(
            "失败清单（可手动下载）:",
            failed
        )


if __name__ == "__main__":
    main()
