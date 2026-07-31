from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any

from app.services.llm import (
    LLMAPIError,
    LLMConfigError,
    LLMClient,
    LLMTimeoutError,
    build_non_thinking_extra_body,
    get_llm_client,
)


IMAGE_ANALYSIS_MODEL = "kimi-k2.6"
ALREADY_SATISFIED = "already satisfied"
VIDEO_LIKE_SUFFIXES = {".gif", ".git"}

IMAGE_TYPE_SCREENSHOT = "screenshot"
IMAGE_TYPE_TABLE = "table"
IMAGE_TYPE_FLOWCHART = "flowchart"
DEFAULT_IMAGE_ANALYSIS_WORKERS = 3
MAX_IMAGE_ANALYSIS_WORKERS = 5
VISION_UPLOAD_MAX_ATTEMPTS = max(
    1,
    int(os.getenv("VISION_UPLOAD_MAX_ATTEMPTS", "3")),
)
VISION_UPLOAD_RETRY_BASE_SECONDS = max(
    0.0,
    float(os.getenv("VISION_UPLOAD_RETRY_BASE_SECONDS", "0.75")),
)


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
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[img_parser][{timestamp}] {message}", flush=True)


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
    image_path = Path(item["path"])
    prompt = _build_prompt(
        context,
        group_position=group_position,
        group_size=group_size,
    )
    try:
        response = request_multimodal_text(
            client,
            prompt=prompt,
            image_bytes=image_path.read_bytes(),
            content_type=mimetypes.guess_type(image_path)[0] or "application/octet-stream",
            model=model,
            max_tokens=1200,
            json_mode=True,
            purpose=f"image analysis: {image_path.name}",
        )
    except LLMTimeoutError:
        # Retrying the same image without JSON mode does not address a network
        # or provider input timeout and used to double the wall-clock delay.
        raise
    except LLMAPIError as exc:
        if not _looks_like_json_mode_error(exc):
            raise
        _log(
            "vision JSON mode unsupported; retrying as plain text: "
            f"image={image_path.name}; error={type(exc).__name__}: {exc}"
        )
        response = request_multimodal_text(
            client,
            prompt=prompt,
            image_bytes=image_path.read_bytes(),
            content_type=mimetypes.guess_type(image_path)[0] or "application/octet-stream",
            model=model,
            max_tokens=1200,
            purpose=f"image analysis fallback: {image_path.name}",
        )
    return _parse_analysis(response)


def request_multimodal_text(
    client: LLMClient,
    *,
    prompt: str,
    image_bytes: bytes,
    content_type: str = "image/jpeg",
    model: str | None = IMAGE_ANALYSIS_MODEL,
    max_tokens: int = 1200,
    read_timeout: float | None = None,
    json_mode: bool = False,
    purpose: str = "multimodal request",
    image_transport: str = "auto",
) -> str:
    """Call the shared, non-streaming vision transport.

    Moonshot requests use its temporary file API by default.  This avoids
    embedding a large base64 image in the chat JSON body, a common source of
    provider-side ``input timeout`` failures.  Other OpenAI-compatible
    providers keep using a Data URL.
    """

    transport = _resolve_multimodal_transport(client, image_transport)
    file_id: str | None = None
    image_url: str
    request_started = time.perf_counter()
    if transport == "file":
        upload_started = time.perf_counter()
        _log(
            "vision upload start: "
            f"purpose={purpose}; image_kb={len(image_bytes) / 1024:.1f}; "
            f"content_type={content_type}; max_attempts={VISION_UPLOAD_MAX_ATTEMPTS}"
        )
        for attempt in range(1, VISION_UPLOAD_MAX_ATTEMPTS + 1):
            try:
                file_id = client.upload_file(
                    filename=f"vision-input{mimetypes.guess_extension(content_type) or '.bin'}",
                    content=image_bytes,
                    purpose="image",
                    content_type=content_type,
                    read_timeout=min(60.0, read_timeout or 60.0),
                )
                break
            except (AttributeError, LLMAPIError, OSError) as exc:
                retryable = _is_retryable_vision_upload_error(exc)
                final_attempt = attempt >= VISION_UPLOAD_MAX_ATTEMPTS
                _log(
                    "vision upload failed: "
                    f"purpose={purpose}; attempt={attempt}/{VISION_UPLOAD_MAX_ATTEMPTS}; "
                    f"retryable={retryable}; "
                    f"elapsed={time.perf_counter() - upload_started:.2f}s; "
                    f"error_kind={_vision_error_kind(exc)}; "
                    f"root_error={_root_error_summary(exc)}; "
                    f"diagnostic={_vision_upload_diagnostic(exc)}; "
                    f"error={type(exc).__name__}: {exc}"
                )
                if final_attempt or not retryable:
                    raise
                delay = VISION_UPLOAD_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                _log(
                    "vision upload retry scheduled: "
                    f"purpose={purpose}; next_attempt={attempt + 1}; "
                    f"delay={delay:.2f}s"
                )
                if delay > 0:
                    time.sleep(delay)
        if not file_id:
            raise LLMAPIError(
                "Vision upload returned no file ID after "
                f"{VISION_UPLOAD_MAX_ATTEMPTS} attempts"
            )
        _log(
            "vision upload complete: "
            f"purpose={purpose}; elapsed={time.perf_counter() - upload_started:.2f}s"
        )
        image_url = f"ms://{file_id}"
    else:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        image_url = f"data:{content_type};base64,{encoded}"

    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": image_url},
        },
    ]
    selected_model = model or getattr(getattr(client, "settings", None), "model", "")
    base_url = str(getattr(getattr(client, "settings", None), "base_url", ""))
    extra_body = build_non_thinking_extra_body(
        model=selected_model,
        base_url=base_url,
    )
    if json_mode:
        extra_body["response_format"] = {"type": "json_object"}
    _log(
        "vision request start: "
        f"purpose={purpose}; transport={transport}; image_kb={len(image_bytes) / 1024:.1f}; "
        f"prompt_chars={len(prompt)}; max_output_tokens={max_tokens}; "
        f"read_timeout={read_timeout}; json_mode={json_mode}; "
        f"model={selected_model or '<default>'}; "
        f"thinking_mode={'disabled' if 'thinking' in extra_body or extra_body.get('enable_thinking') is False else 'provider_default'}"
    )
    chat_started = time.perf_counter()
    try:
        response = client.chat(
            [{"role": "user", "content": content}],
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            extra_body=extra_body,
            read_timeout=read_timeout,
            stream=False,
        )
        _log(
            "vision response received: "
            f"purpose={purpose}; transport={transport}; "
            f"model_elapsed={time.perf_counter() - chat_started:.2f}s; "
            f"response_chars={len(response)}"
        )
    except (LLMAPIError, LLMConfigError, OSError, ValueError) as exc:
        _log(
            "vision request failed: "
            f"purpose={purpose}; transport={transport}; "
            f"elapsed={time.perf_counter() - request_started:.2f}s; "
            f"error_kind={_vision_error_kind(exc)}; "
            f"error={type(exc).__name__}: {exc}"
        )
        raise
    finally:
        if file_id:
            cleanup_started = time.perf_counter()
            try:
                client.delete_file(file_id, read_timeout=10.0)
            except (AttributeError, LLMAPIError, OSError) as exc:
                _log(
                    "vision file cleanup failed: "
                    f"purpose={purpose}; error={type(exc).__name__}: {exc}"
                )
            else:
                _log(
                    "vision file cleanup complete: "
                    f"purpose={purpose}; elapsed={time.perf_counter() - cleanup_started:.2f}s"
                )
    _log(
        "vision request complete: "
        f"purpose={purpose}; transport={transport}; "
        f"elapsed={time.perf_counter() - request_started:.2f}s; "
        f"response_chars={len(response)}"
    )
    return response


def _resolve_multimodal_transport(client: LLMClient, configured: str) -> str:
    normalized = str(configured or "auto").strip().lower()
    if normalized not in {"auto", "file", "data_uri"}:
        raise ValueError(f"Unsupported vision image transport: {configured}")
    if normalized != "auto":
        return normalized
    settings = getattr(client, "settings", None)
    base_url = str(getattr(settings, "base_url", "")).lower()
    return "file" if "moonshot" in base_url and hasattr(client, "upload_file") else "data_uri"


def _looks_like_json_mode_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "response_format",
            "json_object",
            "json mode",
            "json schema",
        )
    )


def _vision_error_kind(exc: BaseException) -> str:
    message = str(exc).lower()
    if isinstance(exc, LLMTimeoutError):
        return "client_timeout"
    if "input timeout" in message or "input processing timeout" in message:
        return "provider_input_timeout"
    if "429" in message or "rate limit" in message:
        return "rate_limit"
    if "response_format" in message or "json_object" in message:
        return "response_format"
    if "content is empty" in message or "finish_reason='length'" in message:
        return "output_budget_exhausted"
    if re.search(r"\b5\d\d\b", message):
        return "provider_5xx"
    return "api_error"


def _is_retryable_vision_upload_error(exc: BaseException) -> bool:
    if isinstance(exc, LLMTimeoutError):
        return True
    message = str(exc).lower()
    if any(
        marker in message
        for marker in (
            "10054",
            "connection reset",
            "connection aborted",
            "connection refused",
            "connection closed",
            "remote protocol",
            "server disconnected",
            "failed to upload llm file",
            "temporarily unavailable",
            "timed out",
            "timeout",
            "rate limit",
            "too many requests",
        )
    ):
        return True
    return bool(
        re.search(
            r"(?:returned|status(?:_code)?[=: ]+)\s*(?:429|5\d\d)\b",
            message,
        )
    )


def _root_error_summary(exc: BaseException) -> str:
    root = exc
    visited: set[int] = set()
    while (
        getattr(root, "__cause__", None) is not None
        and id(root) not in visited
    ):
        visited.add(id(root))
        root = root.__cause__  # type: ignore[assignment]
    text = re.sub(r"\s+", " ", str(root)).strip()
    return f"{type(root).__name__}:{text[:180]}"


def _vision_upload_diagnostic(exc: BaseException) -> str:
    message = f"{exc} {_root_error_summary(exc)}".lower()
    if "10054" in message or "connection reset" in message:
        return "peer_or_proxy_reset_connection"
    if "server disconnected" in message or "remote protocol" in message:
        return "provider_or_gateway_closed_connection"
    if "429" in message or "rate limit" in message:
        return "provider_rate_limit"
    if re.search(r"\b5\d\d\b", message):
        return "provider_5xx"
    if "timeout" in message or "timed out" in message:
        return "upload_timeout"
    if "connection refused" in message:
        return "endpoint_or_proxy_refused_connection"
    return "nontransient_or_unknown_upload_error"


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
    except json.JSONDecodeError as exc:
        _log(
            "vision response JSON invalid; using plain-text fallback: "
            f"response_chars={len(response)}; line={exc.lineno}; column={exc.colno}; "
            f"error={exc.msg}"
        )
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
