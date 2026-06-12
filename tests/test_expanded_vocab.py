from __future__ import annotations

from pathlib import Path

import jieba

from app.services.embedding import load_jieba_expanded_vocab
from scripts.vocab.extract_expanded_vocab import (
    extract_vocab_from_text,
    filter_paragraph_lines,
    parse_vocab_response,
    parse_vocab_fallback,
    split_text_for_llm,
)


def test_filter_paragraph_lines_keeps_only_paragraph_text() -> None:
    text = "\n".join(
        [
            "[paragraph] [标题 1] 管理动作频率表",
            "[table] [正文] [{\"字段\":\"值\"}]",
            "[image] [图片] data/processing/demo.png（图片描述）",
            "[paragraph] [正文] 酷学院与一书一课使用说明",
            "[link_ref] [链接] https://example.com（链接描述）",
        ]
    )

    assert filter_paragraph_lines(text) == "管理动作频率表\n酷学院与一书一课使用说明"


def test_split_text_for_llm_keeps_chunks_under_limit() -> None:
    chunks = split_text_for_llm("第一行\n第二行很长\n第三行", max_chars=8)

    assert chunks == ["第一行", "第二行很长", "第三行"]


def test_parse_vocab_response_accepts_json_fence_and_normalizes_pos() -> None:
    response = """
```json
[
  {"word": "一书一课", "pos": "nz"},
  {"word": "2026", "pos": "n"},
  {"word": "迈拓控股", "pos": "nt"},
  {"word": "奇怪词", "pos": "unknown"}
]
```
"""

    items = parse_vocab_response(response)

    assert [(item.word, item.pos) for item in items] == [
        ("一书一课", "nz"),
        ("迈拓控股", "nt"),
        ("奇怪词", "nz"),
    ]


def test_parse_vocab_fallback_handles_non_json_lines() -> None:
    response = """
- 一书一课,nz
- 迈拓控股：nt
{"word": "科学上网", "pos": "l"},
"""

    items = parse_vocab_fallback(response)

    assert [(item.word, item.pos) for item in items] == [
        ("一书一课", "nz"),
        ("科学上网", "l"),
        ("迈拓控股", "nt"),
    ]


def test_parse_vocab_fallback_returns_empty_for_truncated_array() -> None:
    assert parse_vocab_fallback("[") == []


def test_extract_vocab_from_text_retries_after_truncated_json() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0
            self.extra_body = None

        def chat(self, messages, **kwargs):
            self.calls += 1
            self.extra_body = kwargs.get("extra_body")
            if self.calls == 1:
                return "["
            return '[{"word":"酷学院","pos":"nz"}]'

    client = FakeClient()

    items = extract_vocab_from_text(client, Path("demo.txt"), "酷学院说明", retries=1)

    assert client.calls == 2
    assert client.extra_body == {"enable_thinking": False}
    assert [(item.word, item.pos) for item in items] == [("酷学院", "nz")]


def test_load_jieba_expanded_vocab_adds_custom_words(tmp_path) -> None:
    vocab_file = tmp_path / "expanded_vocab.csv"
    vocab_file.write_text("word,pos\n酷学院,nz\n", encoding="utf-8")

    loaded = load_jieba_expanded_vocab(vocab_file, freq=100000)

    assert loaded == 1
    assert "酷学院" in list(jieba.cut("酷学院登录入口", cut_all=False))
