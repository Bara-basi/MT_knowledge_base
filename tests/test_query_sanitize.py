from __future__ import annotations

from app.api.v1.query import sanitize_question_for_n8n


def test_sanitize_question_unescapes_literal_quotes() -> None:
    question = r'迈拓提到的SAFE具体指什么？他们在\"先进\"方面有哪些具体举措？'

    assert (
        sanitize_question_for_n8n(question)
        == '迈拓提到的SAFE具体指什么？他们在"先进"方面有哪些具体举措？'
    )


def test_sanitize_question_collapses_literal_control_escapes() -> None:
    question = r"第一行\n第二行\t第三段"

    assert sanitize_question_for_n8n(question) == "第一行 第二行 第三段"


def test_sanitize_question_removes_invisible_control_characters() -> None:
    question = "迈拓\u200b提到\x00的SAFE\u2028具体指什么？"

    assert sanitize_question_for_n8n(question) == "迈拓提到的SAFE 具体指什么？"
