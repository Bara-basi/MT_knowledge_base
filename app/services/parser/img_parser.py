from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Any

from app.services.llm import LLMAPIError, LLMConfigError, LLMClient, get_llm_client


IMAGE_ANALYSIS_MODEL = "Pro/moonshotai/Kimi-K2.6"
ALREADY_SATISFIED = "already satisfied"
VIDEO_LIKE_SUFFIXES = {".gif", ".git"}

IMAGE_TYPE_SCREENSHOT = "screenshot"
IMAGE_TYPE_TABLE = "table"
IMAGE_TYPE_FLOWCHART = "flowchart"
DEFAULT_IMAGE_ANALYSIS_WORKERS = 3
MAX_IMAGE_ANALYSIS_WORKERS = 5


def enrich_image_descriptions(
    items: list[dict[str, str]],
    *,
    document_title: str = "",
    llm_client: LLMClient | None = None,
    model: str = IMAGE_ANALYSIS_MODEL,
    max_concurrency: int = DEFAULT_IMAGE_ANALYSIS_WORKERS,
) -> list[dict[str, str]]:
    """Classify and describe image items with bounded concurrent LLM requests."""
    started_at = time.perf_counter()
    client = _get_optional_client(llm_client)
    tasks = _build_analysis_tasks(items)
    for task in tasks:
        if document_title:
            task["context"]["document_title"] = document_title
    _log(f"image analysis tasks: {len(tasks)}")

    if client is None:
        reason = "LLM client unavailable"
        _log(f"image analysis warning: {reason}; using fallback image descriptions")
        for task in tasks:
            _apply_fallback_description(items[task["item_index"]], task["fallback_text"], reason=reason)
        _log(f"finished fallback image descriptions ({time.perf_counter() - started_at:.2f}s)")
        return items

    api_tasks = []
    for task in tasks:
        item = items[task["item_index"]]
        if _is_video_like_item(item):
            _apply_fallback_description(item, task["fallback_text"])
            _log(f"skipped video-like image: {item.get('path') or item.get('text')}")
        else:
            api_tasks.append(task)

    worker_count = _resolve_worker_count(max_concurrency, len(api_tasks))
    _log(f"image API tasks: {len(api_tasks)}, workers={worker_count}")
    if worker_count <= 1:
        for done_count, task in enumerate(api_tasks, start=1):
            _analyze_and_apply_image(items, task, client=client, model=model)
            item = items[task["item_index"]]
            _log(f"image analysis progress: {done_count}/{len(api_tasks)} {item.get('path') or item.get('text')}")
        _log_all_satisfied_warning(items, api_tasks)
        _log(f"finished image analysis ({time.perf_counter() - started_at:.2f}s)")
        return items

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_task = {
            executor.submit(
                _analyze_image,
                dict(items[task["item_index"]]),
                task["context"],
                client=client,
                model=model,
                group_position=task["group_position"],
                group_size=task["group_size"],
            ): task
            for task in api_tasks
        }
        for done_count, future in enumerate(as_completed(future_to_task), start=1):
            task = future_to_task[future]
            item = items[task["item_index"]]
            try:
                analysis = future.result()
            except (LLMAPIError, LLMConfigError, OSError, ValueError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                _apply_fallback_description(item, task["fallback_text"], reason=reason)
                _log(
                    f"image analysis fallback: {done_count}/{len(api_tasks)} "
                    f"{item.get('path') or item.get('text')} reason={reason}"
                )
                continue
            _apply_analysis(item, analysis, task["fallback_text"])
            _log(f"image analysis progress: {done_count}/{len(api_tasks)} {item.get('path') or item.get('text')}")

    _log_all_satisfied_warning(items, api_tasks)
    _log(f"finished image analysis ({time.perf_counter() - started_at:.2f}s)")
    return items


def _log(message: str) -> None:
    print(f"[img_parser] {message}", flush=True)


def _log_all_satisfied_warning(items: list[dict[str, str]], tasks: list[dict[str, Any]]) -> None:
    if not tasks:
        return
    analyzed_items = [items[task["item_index"]] for task in tasks]
    if analyzed_items and all(item.get("description", "").strip().lower() == ALREADY_SATISFIED for item in analyzed_items):
        _log("image analysis warning: all analyzed images returned already satisfied; check visual model output quality")


def _build_analysis_tasks(items: list[dict[str, str]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for group in _build_image_groups(items):
        context = _build_context(items, group[0])
        for position, item_index in enumerate(group, start=1):
            tasks.append(
                {
                    "item_index": item_index,
                    "context": context,
                    "fallback_text": context["nearest_body_text"],
                    "group_position": position,
                    "group_size": len(group),
                }
            )
    return tasks


def _resolve_worker_count(max_concurrency: int, task_count: int) -> int:
    if task_count <= 0:
        return 0
    return max(1, min(max_concurrency, MAX_IMAGE_ANALYSIS_WORKERS, task_count))


def _analyze_and_apply_image(
    items: list[dict[str, str]],
    task: dict[str, Any],
    *,
    client: LLMClient,
    model: str,
) -> None:
    item = items[task["item_index"]]
    try:
        analysis = _analyze_image(
            item,
            task["context"],
            client=client,
            model=model,
            group_position=task["group_position"],
            group_size=task["group_size"],
        )
    except (LLMAPIError, LLMConfigError, OSError, ValueError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        _apply_fallback_description(item, task["fallback_text"], reason=reason)
        _log(f"image analysis fallback: {item.get('path') or item.get('text')} reason={reason}")
        return
    _apply_analysis(item, analysis, task["fallback_text"])


def _get_optional_client(llm_client: LLMClient | None) -> LLMClient | None:
    if llm_client is not None:
        return llm_client
    try:
        return get_llm_client()
    except LLMConfigError:
        return None


def _build_image_groups(items: list[dict[str, str]]) -> list[list[int]]:
    groups: list[list[int]] = []
    current_group: list[int] = []

    for index, item in enumerate(items):
        if item.get("type") == "image":
            current_group.append(index)
        elif current_group:
            groups.append(current_group)
            current_group = []

    if current_group:
        groups.append(current_group)
    return groups


def _build_context(items: list[dict[str, str]], image_index: int) -> dict[str, str]:
    document_title = ""
    heading_path: list[str] = []
    nearest_body_text = ""

    for item in items[:image_index]:
        if item.get("type") == "image":
            continue

        text = item.get("text", "").strip()
        if not text:
            continue

        style = item.get("style", "")
        if _is_title_style(style) and not document_title:
            document_title = text
        elif _is_heading_style(style):
            heading_path.append(text)
        elif _is_body_style(style) or item.get("type", "").startswith("table"):
            nearest_body_text = text

    return {
        "document_title": document_title,
        "heading_path": " -> ".join(heading_path[-4:]),
        "nearest_body_text": nearest_body_text,
    }


def _is_video_like_item(item: dict[str, str]) -> bool:
    path = item.get("path") or item.get("text") or ""
    return Path(path).suffix.lower() in VIDEO_LIKE_SUFFIXES


def _is_title_style(style: str) -> bool:
    return style in {"标题", "Title"}


def _is_heading_style(style: str) -> bool:
    normalized = style.strip().lower()
    return style.startswith("标题") or normalized.startswith("heading")


def _is_body_style(style: str) -> bool:
    return style in {"正文", "Normal"}


def _apply_fallback_description(
    item: dict[str, str],
    fallback_text: str,
    *,
    reason: str | None = None,
) -> None:
    item["image_type"] = IMAGE_TYPE_SCREENSHOT
    if reason:
        item["description"] = (
            f"图片解析失败：{reason}。"
            + (f" 上下文参考：{fallback_text}" if fallback_text else "")
        )
        return

    item["description"] = ALREADY_SATISFIED if fallback_text else "图片位于教程上下文中，暂无可用正文描述。"


def _analyze_image(
    item: dict[str, str],
    context: dict[str, str],
    *,
    client: LLMClient,
    model: str,
    group_position: int,
    group_size: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": _build_prompt(context, group_position=group_position, group_size=group_size),
        },
        {
            "type": "image_url",
            "image_url": {"url": _image_to_data_url(Path(item["path"]))},
        },
    ]
    # print(content[0].get("text"))
    messages = [{"role": "user", "content": content}]
    try:
        response = client.chat(
            messages,
            model=model,
            temperature=0,
            max_tokens=1200,
            extra_body={
                "response_format": {"type": "json_object"},
                "enable_thinking": False,
            },
        )
    except LLMAPIError:
        response = client.chat(
            messages,
            model=model,
            temperature=0,
            max_tokens=1200,
        )
    return _parse_analysis(response)


def _build_prompt(context: dict[str, str], *, group_position: int, group_size: int) -> str:
    return f"""
你是企业内部知识库的教程图片解析器。当前请求只包含一张图片；它是原连续图片组中的第 {group_position} 张，共 {group_size} 张。

上下文元数据：
- 文件名称：{context["document_title"] or "未知"}
- 当前文件分块：{context["heading_path"] or "未知"}
- 最近一条上文正文：{context["nearest_body_text"] or "无"}

请先判断图片类型，再按类型抽取信息：
1. screenshot：操作界面截图。若图片只是上文教程步骤的截图说明，或上文已经充分覆盖图片信息，description 必须输出 already satisfied，不要复述上文。只有图片补充了上文没有表达的关键按钮、界面位置、填写项或注意事项时，才写具体 description。
2. table：图片主体是表格。必须尽可能只字不差转成 JSON，放入 table_data；不要总结、不要改写、不要遗漏文字,每行数据都应该带有列头信息，如果有表名，则记录到table_name中。
3. flowchart：流程图。description 用自然语言描述原文中的主干流程，关注关键阶段/步骤，复杂细节可省略,流程说明不宜过长，中文字符控制在200字以内，其它字符控制在500字以内。

输出必须是 JSON 对象，不要 Markdown，不要代码块，不要额外解释：
{{
  "image_type": "screenshot|table|flowchart",
  "description": "描述文本或 already satisfied",
  "table_name":"表名,没有则填null",
  "table_data": null
}}
""".strip()


def _apply_analysis(item: dict[str, str], analysis: dict[str, Any], fallback_text: str) -> None:
    image_type = str(analysis.get("image_type") or IMAGE_TYPE_SCREENSHOT).strip().lower()

    if image_type == IMAGE_TYPE_TABLE:
        item["original_type"] = item.get("type", "")
        item["original_path"] = item.get("path", "")
        item.pop("path", None)
        item["type"] = "image_table"
        item["style"] = "图片表格"
        item["image_type"] = IMAGE_TYPE_TABLE
        table_name = _normalize_table_name(analysis.get("table_name"))
        if table_name:
            item["table_name"] = table_name
        item["text"] = _format_table_data(analysis.get("table_data"), fallback_text)
        return

    item["image_type"] = IMAGE_TYPE_FLOWCHART if image_type == IMAGE_TYPE_FLOWCHART else IMAGE_TYPE_SCREENSHOT
    item["description"] = _normalize_description(analysis.get("description"), fallback_text)


def _normalize_description(description: Any, fallback_text: str) -> str:
    text = str(description or "").strip()
    if text.lower() == ALREADY_SATISFIED:
        return ALREADY_SATISFIED
    if _same_meaning_as_context(text, fallback_text):
        return ALREADY_SATISFIED
    return text or fallback_text or "图片位于教程上下文中，暂无可用正文描述。"


def _same_meaning_as_context(description: str, context: str) -> bool:
    normalized_description = _normalize_for_compare(description)
    normalized_context = _normalize_for_compare(context)
    if not normalized_description or not normalized_context:
        return False
    return normalized_description == normalized_context


def _normalize_for_compare(text: str) -> str:
    return "".join(str(text).split()).strip("。；;，,：:")


def _format_table_data(table_data: Any, fallback_text: str) -> str:
    if table_data not in (None, ""):
        return json.dumps(table_data, ensure_ascii=False, separators=(",", ":"))
    return fallback_text or "图片表格未能结构化提取。"


def _normalize_table_name(table_name: Any) -> str:
    if table_name in (None, ""):
        return ""
    if isinstance(table_name, str):
        text = table_name.strip()
        return "" if text.lower() == "null" else text
    return json.dumps(table_name, ensure_ascii=False, separators=(",", ":"))


def _image_to_data_url(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _parse_analysis(response: str) -> dict[str, Any]:
    cleaned = _extract_json_payload(response)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "image_type": IMAGE_TYPE_SCREENSHOT,
            "description": _strip_model_artifacts(response).strip(),
            "table_data": None,
        }

    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        raise ValueError(f"Image analysis response is not a JSON object: {response}")
    return parsed


def _strip_json_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_payload(text: str) -> str:
    stripped = _strip_json_code_fence(_strip_model_artifacts(text))
    direct_json = _slice_balanced_json(stripped)
    if direct_json is not None:
        return direct_json
    return stripped


def _strip_model_artifacts(text: str) -> str:
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", text)
    cleaned = re.sub(r"(?is).*?</think>", "", cleaned)
    cleaned = re.sub(r"(?im)^```(?:json)?\s*$", "", cleaned)
    return cleaned.strip()


def _slice_balanced_json(text: str) -> str | None:
    for start in (index for index, char in enumerate(text) if char in "[{"):
        candidate = _balanced_json_from(text, start)
        if candidate is not None:
            return candidate
    return None


def _balanced_json_from(text: str, start: int) -> str | None:
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    stack = [closing]
    in_string = False
    escaped = False

    for index in range(start + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[start : index + 1].strip()

    return None
