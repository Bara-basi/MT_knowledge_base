from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.llm import LLMClient


DEFAULT_OUTPUT_FILE = Path("data") / "vocab" / "expanded_vocab.csv"
DEFAULT_MAX_CHARS = 6000
DEFAULT_RETRIES = 2
DEFAULT_POS = "nz"
PARAGRAPH_LINE_PATTERN = re.compile(
    r"^\[paragraph]\s+\[(?P<style>[^\]]+)]\s*(?P<text>.*)$"
)
ALLOWED_POS_TAGS = {
    "n",
    "nr",
    "ns",
    "nt",
    "nz",
    "eng",
    "vn",
    "v",
    "l",
}


@dataclass(frozen=True)
class VocabItem:
    word: str
    pos: str


def main() -> None:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "Extract an expanded enterprise-specific jieba vocabulary from txt files with an LLM. "
            "This only updates the CSV vocabulary; it does not rebuild BM25."
        ),
    )
    parser.add_argument(
        "input_path",
        help="A txt file or a folder containing txt files.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help=f"Vocabulary CSV output path. Default: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan folders recursively. Enabled by default.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum characters sent to the LLM per txt file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional LLM model override.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Retry count per text chunk when the LLM response cannot be parsed.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model thinking mode. Disabled by default for faster vocab extraction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print extracted terms without writing the output CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    txt_files = find_txt_files(input_path, recursive=args.recursive)
    if not txt_files:
        raise SystemExit(f"No txt files found under: {input_path}")

    client = LLMClient()
    extracted: list[VocabItem] = []
    for index, txt_file in enumerate(txt_files, start=1):
        raw_text = txt_file.read_text(encoding="utf-8", errors="replace")
        text = filter_paragraph_lines(raw_text)
        print(
            f"[{index}/{len(txt_files)}] Extracting vocab from {txt_file} "
            f"(paragraph_chars={len(text)})",
            flush=True,
        )
        if not text.strip():
            print("  skipped=empty paragraph text", flush=True)
            continue
        chunks = split_text_for_llm(text, max_chars=max(args.max_chars, 1))
        for chunk_index, chunk in enumerate(chunks, start=1):
            print(
                f"  chunk={chunk_index}/{len(chunks)} chars={len(chunk)}",
                flush=True,
            )
            items = extract_vocab_from_text(
                client,
                txt_file,
                chunk,
                model=args.model,
                retries=max(args.retries, 0),
                enable_thinking=args.enable_thinking,
            )
            print(f"    extracted={len(items)}", flush=True)
            extracted.extend(items)

    output_file = Path(args.output)
    merged = merge_vocab(load_existing_vocab(output_file), extracted)

    if args.dry_run:
        for item in merged:
            print(f"{item.word},{item.pos}")
        return

    save_vocab(output_file, merged)
    print(f"Saved {len(merged)} terms to {output_file}", flush=True)
    print("BM25 was not rebuilt. Review the CSV before reindexing.", flush=True)


def find_txt_files(input_path: Path, *, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".txt":
            raise ValueError(f"Input file must be a .txt file: {input_path}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Input path must be a txt file or folder: {input_path}")

    iterator = input_path.rglob("*.txt") if recursive else input_path.glob("*.txt")
    return sorted(path for path in iterator if path.is_file())


def filter_paragraph_lines(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = PARAGRAPH_LINE_PATTERN.match(line)
        if match is None:
            continue
        paragraph = match.group("text").strip()
        if paragraph:
            lines.append(paragraph)
    return "\n".join(lines)


def split_text_for_llm(text: str, *, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for line in text.splitlines():
        line_length = len(line) + 1
        if current and current_length + line_length > max_chars:
            chunks.append("\n".join(current).strip())
            current = []
            current_length = 0
        if line_length > max_chars:
            chunks.extend(_split_long_line(line, max_chars=max_chars))
            continue
        current.append(line)
        current_length += line_length

    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _split_long_line(line: str, *, max_chars: int) -> list[str]:
    return [
        line[start : start + max_chars].strip()
        for start in range(0, len(line), max_chars)
        if line[start : start + max_chars].strip()
    ]


def extract_vocab_from_text(
    client: LLMClient,
    txt_file: Path,
    text: str,
    *,
    model: str | None = None,
    retries: int = DEFAULT_RETRIES,
    enable_thinking: bool = False,
) -> list[VocabItem]:
    messages = _build_messages(txt_file, text)
    last_response = ""
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        response = client.chat(
            messages,
            model=model,
            temperature=0.0 if attempt else 0.1,
            max_tokens=1200,
            extra_body={"enable_thinking": enable_thinking},
        )
        last_response = response
        try:
            return parse_vocab_response(response)
        except ValueError as exc:
            last_error = exc
            fallback_items = parse_vocab_fallback(response)
            if fallback_items:
                print(
                    f"    warning=parse fallback used items={len(fallback_items)} "
                    f"reason={exc}",
                    flush=True,
                )
                return fallback_items
            if attempt >= retries:
                break
            print(
                f"    warning=parse failed attempt={attempt + 1}/{retries + 1}: {exc}",
                flush=True,
            )
            messages = _build_retry_messages(last_response)
            time.sleep(min(2 ** attempt, 5))

    print(
        "    warning=skip chunk after parse failures "
        f"reason={last_error} response_preview={last_response[:200]!r}",
        flush=True,
    )
    return []


def _build_messages(txt_file: Path, text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是企业内部知识库的分词词表构建助手。"
                "你的任务是从文本中提取适合加入 jieba 自定义词典的词。"
                "必须只输出合法 JSON 数组。"
            ),
        },
        {
            "role": "user",
            "content": _build_prompt(txt_file, text),
        },
    ]


def _build_retry_messages(response: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "你是 JSON 修复器。必须只输出合法 JSON 数组，不要解释。",
        },
        {
            "role": "user",
            "content": (
                "下面内容不是合法 JSON 数组。请将其中可识别的词汇修复为格式 "
                '[{"word":"词","pos":"nz"}]。如果无法识别任何词，返回 []。\n\n'
                f"{response[:4000]}"
            ),
        },
    ]


def _build_prompt(txt_file: Path, text: str) -> str:
    return f"""
请从下面这份企业内部知识库 txt 中提取适合加入 jieba 用户词典的高价值词汇。

提取对象：
1. 专有实体名词：软件名、系统名、公司/组织名、内部项目名、内部代号、产品名、平台名。
2. 别名、内部说法、行业黑话或固定短语，例如“科学上网”。

不要提取：
1. 普通通用名词、单字词、没有独立检索价值的短词。
2. 过长句子、完整说明句、URL、文件路径、日期、纯数字。
3. 已经是很自然的普通分词结果且不具有企业内部含义的词。

请按 jieba 词性标注，优先使用：
- nz：其他专名、软件名、产品名、内部代号
- nt：组织机构名
- n：普通名词
- eng：英文名或英文缩写
- l：固定短语/习惯说法

只返回 JSON 数组，不要返回解释。格式如下：
[
  {{"word": "一书一课", "pos": "nz"}},
  {{"word": "迈拓控股", "pos": "nt"}},
  {{"word": "科学上网", "pos": "l"}}
]

文件：{txt_file}

文本：
{text}
""".strip()


def parse_vocab_response(response: str) -> list[VocabItem]:
    payload = _extract_json_array(response)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {response}") from exc

    if not isinstance(data, list):
        raise ValueError(f"LLM response must be a JSON array: {response}")

    items: list[VocabItem] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        word = normalize_word(item.get("word"))
        pos = normalize_pos(item.get("pos"))
        if word:
            items.append(VocabItem(word=word, pos=pos))
    return items


def parse_vocab_fallback(response: str) -> list[VocabItem]:
    items: list[VocabItem] = []
    for raw_line in response.splitlines():
        line = raw_line.strip().strip("-*• \t")
        if not line or line in {"[", "]"}:
            continue
        line = line.rstrip(",;；")
        item = _parse_fallback_line(line)
        if item is not None:
            items.append(item)
    return merge_vocab([], items)


def _parse_fallback_line(line: str) -> VocabItem | None:
    json_like = _parse_json_object_line(line)
    if json_like is not None:
        return json_like

    for separator in (",", "，", "\t", "|", "：", ":"):
        if separator not in line:
            continue
        left, right = [part.strip() for part in line.split(separator, 1)]
        word = normalize_word(left.strip('"“”'))
        pos = normalize_pos(right.strip('"“”'))
        if word:
            return VocabItem(word=word, pos=pos)

    match = re.match(r"^(?P<word>[\w\u4e00-\u9fff·（）()《》-]{2,})\s+(?P<pos>[a-z]{1,4})$", line)
    if match is not None:
        word = normalize_word(match.group("word"))
        if word:
            return VocabItem(word=word, pos=normalize_pos(match.group("pos")))
    return None


def _parse_json_object_line(line: str) -> VocabItem | None:
    candidate = line.rstrip(",")
    if not candidate.startswith("{") or not candidate.endswith("}"):
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    word = normalize_word(data.get("word"))
    if not word:
        return None
    return VocabItem(word=word, pos=normalize_pos(data.get("pos")))


def _extract_json_array(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped

    match = re.search(r"\[[\s\S]*]", stripped)
    if match is None:
        raise ValueError(f"Cannot find JSON array in LLM response: {text}")
    return match.group(0)


def normalize_word(value: Any) -> str:
    word = str(value or "").strip()
    word = re.sub(r"\s+", " ", word)
    if len(word) < 2:
        return ""
    if re.fullmatch(r"https?://\S+|[\\/:\w.-]+\.(?:txt|docx?|pptx?|xlsx?|pdf)", word):
        return ""
    if re.fullmatch(r"\d+(?:[./-]\d+)*", word):
        return ""
    return word


def normalize_pos(value: Any) -> str:
    pos = str(value or DEFAULT_POS).strip().lower()
    return pos if pos in ALLOWED_POS_TAGS else DEFAULT_POS


def load_existing_vocab(path: Path) -> list[VocabItem]:
    if not path.exists():
        return []

    items: list[VocabItem] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            word = normalize_word(row.get("word") or row.get("词"))
            pos = normalize_pos(row.get("pos") or row.get("词性"))
            if word:
                items.append(VocabItem(word=word, pos=pos))
    return items


def merge_vocab(existing: list[VocabItem], extracted: list[VocabItem]) -> list[VocabItem]:
    merged: dict[str, VocabItem] = {}
    for item in [*existing, *extracted]:
        if item.word not in merged:
            merged[item.word] = item
    return sorted(merged.values(), key=lambda item: item.word)


def save_vocab(path: Path, items: list[VocabItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["word", "pos"])
        writer.writeheader()
        for item in items:
            writer.writerow({"word": item.word, "pos": item.pos})


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
