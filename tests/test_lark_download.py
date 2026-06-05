# -*- coding: utf-8 -*-
"""
独立测试脚本：自动获取 tenant_access_token，并测试是否可以导出并下载飞书新版文档为 docx

测试文档：
https://tmqhw1h9zt.feishu.cn/docx/PBw6dyxsXopGPvxBgyJc8Dy7nWh

使用方式：
1. 安装依赖：
   pip install requests

2. 填写 APP_ID / APP_SECRET

3. 运行：
   python test_feishu_docx_download_tenant.py
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


# =========================
# 需要你填写的配置
# =========================

APP_ID = "cli_aa895b208df9dcdd"
APP_SECRET = "AfESVeY5n9m7By2plKh97g05C7TtbCAZ"

DOC_URL = "https://tmqhw1h9zt.feishu.cn/docx/PBw6dyxsXopGPvxBgyJc8Dy7nWh"

# 新版文档优先导出 docx；也可以改为 pdf 测试
EXPORT_EXTENSION = "docx"

OUTPUT_DIR = "./feishu_download_test_output"

MAX_WAIT_SECONDS = 120
POLL_INTERVAL_SECONDS = 3


# =========================
# 工具函数
# =========================

def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def pretty_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def mask_secret(value: str) -> str:
    if not value:
        return "<未填写>"
    if len(value) <= 10:
        return value[:2] + "***"
    return value[:6] + "..." + value[-4:]


def request_json(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Tuple[int, Dict[str, Any], str]:
    try:
        resp = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=json_body,
            params=params,
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"HTTP 请求异常：{type(e).__name__}: {e}")

    raw_text = resp.text or ""

    try:
        data = resp.json()
    except Exception:
        data = {}

    return resp.status_code, data, raw_text


def parse_feishu_doc_url(url: str) -> Tuple[str, str]:
    log("开始解析飞书文档链接。")
    log(f"原始链接：{url}")

    patterns = [
        (r"/docx/([A-Za-z0-9]+)", "docx"),
        (r"/docs/([A-Za-z0-9]+)", "doc"),
        (r"/sheets/([A-Za-z0-9]+)", "sheet"),
        (r"/base/([A-Za-z0-9]+)", "bitable"),
    ]

    for pattern, file_type in patterns:
        match = re.search(pattern, url)
        if match:
            token = match.group(1)
            log(f"解析成功：token={token}，type={file_type}")
            return token, file_type

    raise ValueError(
        "无法从链接中解析文档 token。请确认链接是否为 /docx/、/docs/、/sheets/ 或 /base/ 类型。"
    )


def get_tenant_access_token(app_id: str, app_secret: str) -> str:
    log("开始获取 tenant_access_token。")

    if not app_id or not app_secret:
        raise ValueError("APP_ID 或 APP_SECRET 为空，请先填写。")

    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/"
    payload = {
        "app_id": app_id,
        "app_secret": app_secret,
    }

    log("请求接口：POST /auth/v3/tenant_access_token/internal/")
    log(f"APP_ID：{mask_secret(app_id)}")
    log(f"APP_SECRET：{mask_secret(app_secret)}")

    http_status, data, raw_text = request_json(
        method="POST",
        url=url,
        headers={"Content-Type": "application/json; charset=utf-8"},
        json_body=payload,
    )

    log(f"获取 tenant_access_token HTTP 状态码：{http_status}")
    log(f"获取 tenant_access_token 响应：\n{pretty_json(data) if data else raw_text}")

    if http_status != 200:
        raise RuntimeError("获取 tenant_access_token 失败：HTTP 状态码不是 200。")

    if data.get("code") != 0:
        raise RuntimeError("获取 tenant_access_token 失败：飞书返回 code != 0。")

    token = data.get("tenant_access_token")
    if not token:
        raise RuntimeError("获取 tenant_access_token 失败：响应中没有 tenant_access_token。")

    log("tenant_access_token 获取成功。")
    log(f"token：{mask_secret(token)}")
    return token


def create_export_task(
    access_token: str,
    doc_token: str,
    doc_type: str,
    export_extension: str,
) -> str:
    log("开始创建飞书云文档导出任务。")
    log("导出任务参数：")
    log(f"- token: {doc_token}")
    log(f"- type: {doc_type}")
    log(f"- file_extension: {export_extension}")

    url = "https://open.feishu.cn/open-apis/drive/v1/export_tasks"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    payload = {
        "file_extension": export_extension,
        "token": doc_token,
        "type": doc_type,
    }

    http_status, data, raw_text = request_json(
        method="POST",
        url=url,
        headers=headers,
        json_body=payload,
    )

    log(f"创建导出任务 HTTP 状态码：{http_status}")
    log(f"创建导出任务响应：\n{pretty_json(data) if data else raw_text}")

    if http_status != 200:
        explain_common_error(http_status, data, step="创建导出任务")
        raise RuntimeError("创建导出任务失败：HTTP 状态码不是 200。")

    if data.get("code") != 0:
        explain_common_error(http_status, data, step="创建导出任务")
        raise RuntimeError("创建导出任务失败：飞书返回 code != 0。")

    ticket = data.get("data", {}).get("ticket")
    if not ticket:
        raise RuntimeError("创建导出任务失败：响应中没有 data.ticket。")

    log(f"创建导出任务成功，ticket={ticket}")
    return ticket


def query_export_task(
    access_token: str,
    ticket: str,
    doc_token: str,
) -> Dict[str, Any]:
    url = f"https://open.feishu.cn/open-apis/drive/v1/export_tasks/{ticket}"
    headers = {
        "Authorization": f"Bearer {access_token}",
    }
    params = {
        "token": doc_token,
    }

    waited = 0

    while waited <= MAX_WAIT_SECONDS:
        log(f"查询导出任务结果，已等待 {waited} 秒。")

        http_status, data, raw_text = request_json(
            method="GET",
            url=url,
            headers=headers,
            params=params,
        )

        log(f"查询导出任务 HTTP 状态码：{http_status}")
        log(f"查询导出任务响应：\n{pretty_json(data) if data else raw_text}")

        if http_status != 200:
            explain_common_error(http_status, data, step="查询导出任务")
            raise RuntimeError("查询导出任务失败：HTTP 状态码不是 200。")

        if data.get("code") != 0:
            explain_common_error(http_status, data, step="查询导出任务")
            raise RuntimeError("查询导出任务失败：飞书返回 code != 0。")

        result = data.get("data", {}).get("result", {})
        job_status = result.get("job_status")

        if job_status == 0:
            file_token = result.get("file_token")
            file_name = result.get("file_name")
            file_extension = result.get("file_extension")
            file_size = result.get("file_size")

            log("导出任务成功。")
            log(f"- file_token: {file_token}")
            log(f"- file_name: {file_name}")
            log(f"- file_extension: {file_extension}")
            log(f"- file_size: {file_size}")

            if not file_token:
                raise RuntimeError("导出任务状态为成功，但响应中没有 file_token。")

            return result

        if job_status == 2:
            log("导出任务失败。")
            log("请重点查看下面这些字段：")
            log(f"- job_status: {job_status}")
            log(f"- job_error_msg: {result.get('job_error_msg')}")
            log(f"- result: {pretty_json(result)}")
            raise RuntimeError("导出任务失败：job_status=2。")

        log(f"导出任务尚未完成，当前 job_status={job_status}，等待后继续查询。")
        time.sleep(POLL_INTERVAL_SECONDS)
        waited += POLL_INTERVAL_SECONDS

    raise TimeoutError(f"导出任务超时：超过 {MAX_WAIT_SECONDS} 秒仍未完成。")


def download_exported_file(
    access_token: str,
    file_token: str,
    file_name: Optional[str],
    export_extension: str,
) -> Path:
    log("开始下载导出后的文件。")
    log(f"导出文件 file_token={file_token}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_file_name = sanitize_filename(file_name or f"feishu_export_result.{export_extension}")
    if "." not in safe_file_name:
        safe_file_name = f"{safe_file_name}.{export_extension}"

    output_path = output_dir / safe_file_name

    url = f"https://open.feishu.cn/open-apis/drive/v1/export_tasks/file/{file_token}/download"
    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=120)
    except requests.RequestException as e:
        raise RuntimeError(f"下载导出文件 HTTP 请求异常：{type(e).__name__}: {e}")

    log(f"下载导出文件 HTTP 状态码：{resp.status_code}")
    log(f"下载导出文件 Content-Type：{resp.headers.get('Content-Type')}")
    log(f"下载导出文件 Content-Length：{resp.headers.get('Content-Length')}")

    if resp.status_code != 200:
        raw_text = resp.text or ""
        log("下载失败响应内容：")
        log(raw_text if raw_text else "<空响应>")
        raise RuntimeError("下载导出文件失败：HTTP 状态码不是 200。")

    content = resp.content
    if not content:
        raise RuntimeError("下载导出文件失败：响应内容为空。")

    content_type = resp.headers.get("Content-Type", "")
    if "application/json" in content_type.lower():
        try:
            data = resp.json()
            log("警告：下载接口返回了 JSON，而不是文件二进制。")
            log(pretty_json(data))
            raise RuntimeError("下载导出文件失败：返回内容是 JSON 错误信息。")
        except ValueError:
            pass

    output_path.write_bytes(content)

    log("文件下载成功。")
    log(f"保存路径：{output_path.resolve()}")
    log(f"文件大小：{len(content)} bytes")

    return output_path


def sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name)
    return name or "feishu_export_result"


def explain_common_error(http_status: int, data: Dict[str, Any], step: str) -> None:
    code = data.get("code")
    msg = data.get("msg")
    error = data.get("error")

    log("错误诊断信息：")
    log(f"- 失败步骤：{step}")
    log(f"- HTTP 状态码：{http_status}")
    log(f"- 飞书 code：{code}")
    log(f"- 飞书 msg：{msg}")
    log(f"- 飞书 error：{pretty_json(error)}")

    if http_status == 401:
        log("可能原因：tenant_access_token 无效、过期，或 Authorization 请求头格式错误。")
        log("建议检查：APP_ID / APP_SECRET 是否正确，应用是否仍然可用。")

    elif http_status == 403 or code == 1069902:
        log("可能原因：应用身份权限不足。")
        log("请重点检查：")
        log("1. 自建应用是否已被添加为该文档的文档应用。")
        log("2. 添加文档应用时是否给了可阅读或可编辑权限。")
        log("3. 应用是否有 docs:document:export 或 drive:export:readonly 权限。")
        log("4. 应用是否已发布，且可用范围包含文档所有者。")
        log("5. 文档权限设置中是否限制了创建副本、打印、下载。")
        log("6. 如果文档位于知识库中，应用是否对该知识库节点有权限。")

    elif code == 1069904:
        log("可能原因：参数错误。")
        log("请检查 token、type、file_extension 是否匹配。")
        log("当前链接是 /docx/ 类型，应使用 type='docx'。")

    else:
        log("未命中内置错误解释。建议复制 log_id 到飞书开放平台排查工具中查看。")


def main() -> int:
    log("=" * 80)
    log("飞书文档导出下载测试开始：tenant_access_token 自动获取模式")
    log("=" * 80)

    log("当前配置：")
    log(f"- DOC_URL: {DOC_URL}")
    log(f"- EXPORT_EXTENSION: {EXPORT_EXTENSION}")
    log(f"- APP_ID: {mask_secret(APP_ID)}")
    log(f"- APP_SECRET: {mask_secret(APP_SECRET)}")
    log(f"- OUTPUT_DIR: {OUTPUT_DIR}")

    try:
        doc_token, doc_type = parse_feishu_doc_url(DOC_URL)

        access_token = get_tenant_access_token(
            app_id=APP_ID.strip(),
            app_secret=APP_SECRET.strip(),
        )

        log("开始执行导出流程。")
        ticket = create_export_task(
            access_token=access_token,
            doc_token=doc_token,
            doc_type=doc_type,
            export_extension=EXPORT_EXTENSION,
        )

        result = query_export_task(
            access_token=access_token,
            ticket=ticket,
            doc_token=doc_token,
        )

        file_token = result.get("file_token")
        file_name = result.get("file_name")

        output_path = download_exported_file(
            access_token=access_token,
            file_token=file_token,
            file_name=file_name,
            export_extension=EXPORT_EXTENSION,
        )

        log("=" * 80)
        log("测试成功：已成功导出并下载飞书文档。")
        log(f"输出文件：{output_path.resolve()}")
        log("=" * 80)
        return 0

    except Exception as e:
        log("=" * 80)
        log("测试失败。")
        log(f"异常类型：{type(e).__name__}")
        log(f"异常信息：{e}")
        log("=" * 80)
        log("排查建议：")
        log("1. 如果获取 tenant_access_token 失败：检查 APP_ID / APP_SECRET 是否正确。")
        log("2. 如果创建导出任务 403 / 1069902：应用身份没有该文档的有效资源权限。")
        log("3. 如果导出任务 job_status=2：检查文档是否允许导出，以及 type/token/file_extension 是否匹配。")
        log("4. 如果下载导出文件失败：检查 drive:file:download 或导出文件下载权限。")
        log("5. 如果你本人能下载但 tenant token 失败，说明用户权限和应用身份权限不等价。")
        return 1


if __name__ == "__main__":
    sys.exit(main())