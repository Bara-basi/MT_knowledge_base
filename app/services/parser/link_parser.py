from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    from app.services.llm import LLMAPIError, LLMConfigError, LLMClient, get_llm_client
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from llm import LLMAPIError, LLMConfigError, LLMClient, get_llm_client


URL_PATTERN = re.compile(r"https?://[^\s<>()，。；;、\"'）】]+", re.IGNORECASE)


def enrich_links(
    items: list[dict[str, str]],
    *,
    llm_client: LLMClient | None = None,
    model: str | None = None,
) -> list[dict[str, str]]:
    """Replace URLs with readable markers and emit link metadata items."""
    client = _get_optional_client(llm_client)
    output: list[dict[str, str]] = []
    nearest_body_text = ""

    for item in items:
        if item.get("type") == "image":
            output.append(item)
            continue

        if item.get("type") == "image_table":
            output.append(item)
            nearest_body_text = item.get("text", "").strip() or nearest_body_text
            continue

        text = item.get("text", "")
        matches = list(URL_PATTERN.finditer(text))
        if not matches:
            output.append(item)
            if _can_be_link_context(item):
                nearest_body_text = text.strip() or nearest_body_text
            continue

        enriched_item, link_items = _replace_urls_with_link_markers(
            item,
            matches,
            nearest_body_text,
            client=client,
            model=model,
        )
        output.append(enriched_item)
        output.extend(link_items)

        remaining_text = URL_PATTERN.sub("", text).strip()
        if remaining_text and _can_be_link_context(item):
            nearest_body_text = remaining_text

    return output


def _get_optional_client(llm_client: LLMClient | None) -> LLMClient | None:
    if llm_client is not None:
        return llm_client
    try:
        return get_llm_client()
    except LLMConfigError:
        return None


def _can_be_link_context(item: dict[str, str]) -> bool:
    style = item.get("style", "")
    return style == "正文" or item.get("type", "").startswith("table")


def _replace_urls_with_link_markers(
    item: dict[str, str],
    matches: list[re.Match[str]],
    nearest_body_text: str,
    *,
    client: LLMClient | None,
    model: str,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    link_items: list[dict[str, str]] = []
    text_parts: list[str] = []
    text = item["text"]
    cursor = 0
    used_descriptions: set[str] = set()

    for match in matches:
        before = text[cursor : match.start()]
        text_parts.append(before)

        url = match.group(0)
        description_context = before.strip() or nearest_body_text
        link_item = _link_item(url, description_context, client=client, model=model)
        description = _unique_description(link_item["description"], used_descriptions)
        link_item["description"] = description
        link_item["text"] = description
        text_parts.append(f"{{{{{description}}}}}")
        link_items.append(link_item)
        cursor = match.end()

    text_parts.append(text[cursor:])
    enriched_item = {**item, "text": "".join(text_parts).strip()}
    return enriched_item, link_items


def _link_item(url: str, context: str, *, client: LLMClient | None, model: str | None) -> dict[str, str]:
    description = _rule_description(context)
    if description is None and client is not None:
        try:
            description = _describe_link(url, context, client=client, model=model)
        except (LLMAPIError, LLMConfigError, OSError, ValueError):
            description = None

    if description is None:
        description = _domain_description(url)

    return {
        "type": "link_ref",
        "style": "链接",
        "text": description,
        "url": url,
        "description": description,
    }


def _unique_description(description: str, used_descriptions: set[str]) -> str:
    if description not in used_descriptions:
        used_descriptions.add(description)
        return description

    index = 2
    while f"{description} {index}" in used_descriptions:
        index += 1
    unique_description = f"{description} {index}"
    used_descriptions.add(unique_description)
    return unique_description


def _describe_link(url: str, context: str, *, client: LLMClient, model: str | None) -> str:
    prompt = f"""
你是企业内部知识库的链接解析器。请根据链接和最接近的上文，为链接生成一个简短、具体、面向知识库检索的中文描述。

要求：
1. 不要泛泛说“这是一个链接”。
2. 描述它在教程中的用途，例如“秀米编辑器网站链接”“飞书订阅号操作手册链接”。
3. 如果上文已明确说明用途，直接提炼上文，不要增加未知信息。
4. 只输出描述本身，不要标点包裹，不要解释。

最近上文：{context or "无"}
链接：{url}
""".strip()

    return client.chat(
        [{"role": "user", "content": prompt}],
        model=model,
        temperature=0,
        max_tokens=120,
    ).strip()


def _fallback_description(url: str, context: str) -> str:
    return _rule_description(context) or _domain_description(url)


def _rule_description(context: str) -> str | None:
    label = _label_before_colon(context)
    if label is None:
        return None
    if label in {"网址", "链接", "URL", "url"}:
        return None
    return f"{label}网站链接" if _looks_like_site_label(label) else label


def _label_before_colon(text: str) -> str | None:
    if not text:
        return None
    if "：" in text:
        label = text.rsplit("：", 1)[0].strip()
    elif ":" in text:
        label = text.rsplit(":", 1)[0].strip()
    else:
        return None
    return label or None


def _domain_description(url: str) -> str:
    domain = re.sub(r"^https?://", "", url, flags=re.IGNORECASE).split("/", 1)[0]
    return f"{domain} 链接"


def _looks_like_site_label(text: str) -> bool:
    return 1 <= len(text) <= 20 and not any(char in text for char in "，。；;,、 ")
