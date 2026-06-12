from __future__ import annotations

import jieba

from app.services.embedding import load_jieba_expanded_vocab
from scripts.vocab.extract_expanded_vocab import parse_vocab_response


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


def test_load_jieba_expanded_vocab_adds_custom_words(tmp_path) -> None:
    vocab_file = tmp_path / "expanded_vocab.csv"
    vocab_file.write_text("word,pos\n酷学院,nz\n", encoding="utf-8")

    loaded = load_jieba_expanded_vocab(vocab_file, freq=100000)

    assert loaded == 1
    assert "酷学院" in list(jieba.cut("酷学院登录入口", cut_all=False))
