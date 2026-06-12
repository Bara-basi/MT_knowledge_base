from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


INVISIBLE_FORMAT_CHARS = {
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u2060",  # word joiner
    "\ufeff",  # byte order mark / zero width no-break space
}
VARIATION_SELECTORS = re.compile(r"[\ufe00-\ufe0f\U000e0100-\U000e01ef]")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f000-\U0001faff"
    "\U00002705"
    "\U00002714"
    "\U00002728"
    "\U000027a1"
    "\U000026a0"
    "\U00002605"
    "\U00002606"
    "]+"
)
REPLACEMENT_CHAR_PATTERN = re.compile("\ufffd+")
COMMON_MOJIBAKE_PATTERN = re.compile(
    r"(?:"
    r"Ã[\x80-\xbf]|"
    r"Â[\x80-\xbf]?|"
    r"â[\x80-\xbf]{1,2}|"
    r"å[\x80-\xbf]{1,2}"
    r")"
)
COMMON_MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€": '"',
    "â€¦": "...",
    "â€“": "-",
    "â€”": "-",
    "Â": "",
}
WHITESPACE_PATTERN = re.compile(r"[ \t\u00a0\u3000]+")
EMPTY_PLACEHOLDER_PATTERN = re.compile(r"^(?:[（(]\s*空\s*[）)]|空|null|none|nan)$", re.IGNORECASE)
SOURCE_SYNC_PATTERN = re.compile(r"^\s*同步自文档\s*[:：]\s*(?:\{\{[^{}]*\}\}|https?://\S*)?\s*$")
IMAGE_PATH_TEXT_PATTERN = re.compile(r"^\s*图片\s*[:：]\s*data[\\/]+processing[\\/].+$", re.IGNORECASE)
EMBEDDED_SOURCE_SYNC_PATTERN = re.compile(r"\s*同步自文档\s*[:：]\s*https?://\S+", re.IGNORECASE)
EMBEDDED_IMAGE_PATH_PATTERN = re.compile(
    r"\s*图片\s*[:：]\s*data[\\/]+processing[\\/].+?\.(?:png|jpe?g|gif|bmp|webp|tiff?)",
    re.IGNORECASE,
)


def clean_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clean parsed document items before chunking.

    The rules are intentionally conservative: remove export artifacts and noisy
    presentation marks while keeping business text, paths, URLs, and metadata.
    """

    output: list[dict[str, Any]] = []
    skip_next_source_link = False

    for item in items:
        cleaned = clean_item(item)
        item_type = str(cleaned.get("type") or "")
        text = str(cleaned.get("text") or "").strip()

        if skip_next_source_link and _is_source_sync_link(cleaned):
            skip_next_source_link = False
            continue
        skip_next_source_link = False

        if _is_source_sync_text(text):
            skip_next_source_link = True
            continue

        if item_type not in {"image", "link", "link_ref"} and _is_image_path_text(text):
            continue

        if item_type not in {"image", "link", "link_ref"} and _is_empty_text(text):
            continue

        output.append(cleaned)

    return output


def clean_item(item: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(item)
    item_type = str(cleaned.get("type") or "")
    for key, value in list(cleaned.items()):
        if not isinstance(value, str):
            continue
        if key in {"path", "url", "source"} or (item_type == "image" and key == "text"):
            cleaned[key] = clean_metadata_text(value)
        elif key == "text" and item_type in {"table", "image_table", "img_table"}:
            cleaned[key] = clean_table_text(value)
        else:
            cleaned[key] = clean_text(value)
    return cleaned


def clean_text(text: Any) -> str:
    value = str(text or "")
    value = _remove_invisible_chars(value)
    value = VARIATION_SELECTORS.sub("", value)
    value = EMOJI_PATTERN.sub("", value)
    value = REPLACEMENT_CHAR_PATTERN.sub("", value)
    value = _replace_common_mojibake(value)
    value = COMMON_MOJIBAKE_PATTERN.sub("", value)
    value = EMBEDDED_SOURCE_SYNC_PATTERN.sub("", value)
    value = EMBEDDED_IMAGE_PATH_PATTERN.sub("", value)
    value = re.sub(r"(?<=\s)[\"'.-]+(?=\s|$)", "", value)
    value = WHITESPACE_PATTERN.sub(" ", value)
    value = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", value)
    value = re.sub(r"([（(【\[])\s+", r"\1", value)
    value = re.sub(r"\s+([）)】\]])", r"\1", value)
    return value.strip()


def clean_metadata_text(text: Any) -> str:
    value = str(text or "")
    value = _remove_invisible_chars(value)
    value = VARIATION_SELECTORS.sub("", value)
    value = REPLACEMENT_CHAR_PATTERN.sub("", value)
    value = _replace_common_mojibake(value)
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def clean_table_text(text: str) -> str:
    raw_text = _remove_invisible_chars(str(text or ""))
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return clean_text(raw_text)

    cleaned = _clean_json_value(parsed)
    return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))


def _clean_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        return [_clean_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            clean_text(key): _clean_json_value(item)
            for key, item in value.items()
            if clean_text(key)
        }
    return value


def _remove_invisible_chars(text: str) -> str:
    chars: list[str] = []
    for char in text:
        if char in INVISIBLE_FORMAT_CHARS:
            continue
        category = unicodedata.category(char)
        if category in {"Cc", "Cs", "Co", "Cn"} and char not in {"\n", "\r", "\t"}:
            continue
        chars.append(char)
    return "".join(chars)


def _replace_common_mojibake(text: str) -> str:
    for old, new in COMMON_MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def _is_source_sync_text(text: str) -> bool:
    return bool(SOURCE_SYNC_PATTERN.match(text))


def _is_source_sync_link(item: dict[str, Any]) -> bool:
    if str(item.get("type") or "") not in {"link", "link_ref"}:
        return False
    description = str(item.get("description") or item.get("text") or "")
    return "同步自文档" in description


def _is_image_path_text(text: str) -> bool:
    return bool(IMAGE_PATH_TEXT_PATTERN.match(text))


def _is_empty_text(text: str) -> bool:
    stripped = text.strip()
    return not stripped or bool(EMPTY_PLACEHOLDER_PATTERN.match(stripped))
