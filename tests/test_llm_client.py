from __future__ import annotations

import pytest

from app.services import llm as llm_module
from app.services.llm import (
    LLMAPIError,
    LLMClient,
    LLMSettings,
    build_non_thinking_extra_body,
)


def _client() -> LLMClient:
    return LLMClient(
        LLMSettings(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
        )
    )


def test_moonshot_kimi_uses_official_non_thinking_parameter() -> None:
    assert build_non_thinking_extra_body(
        model="kimi-k2.5",
        base_url="https://api.moonshot.cn/v1",
    ) == {"thinking": {"type": "disabled"}}
    assert build_non_thinking_extra_body(
        model="deepseek-v3",
        base_url="https://api.siliconflow.cn/v1",
    ) == {"enable_thinking": False}


def test_chat_rejects_empty_content_with_finish_metadata(monkeypatch) -> None:
    client = _client()
    monkeypatch.setattr(
        client,
        "_post",
        lambda *_args, **_kwargs: {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": "",
                        "reasoning_content": "internal reasoning",
                    },
                }
            ]
        },
    )

    with pytest.raises(
        LLMAPIError,
        match=r"finish_reason='length'.*reasoning_length=18",
    ):
        client.chat([{"role": "user", "content": "test"}])


def test_chat_accepts_openai_text_content_blocks(monkeypatch) -> None:
    client = _client()
    monkeypatch.setattr(
        client,
        "_post",
        lambda *_args, **_kwargs: {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": [
                            {"type": "text", "text": '{"ok":'},
                            {"type": "output_text", "text": "true}"},
                        ]
                    },
                }
            ]
        },
    )

    assert client.chat([{"role": "user", "content": "test"}]) == '{"ok":true}'


def test_chat_streams_sse_content_incrementally(monkeypatch) -> None:
    captured_payload: dict = {}

    class FakeResponse:
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"{\\"ok\\":"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"true}"}}]}'
            yield "data: [DONE]"

    class FakeHTTPClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def stream(self, _method, _url, *, json, headers):
            del headers
            captured_payload.update(json)
            return FakeResponse()

    monkeypatch.setattr(llm_module.httpx, "Client", FakeHTTPClient)
    client = _client()

    result = client.chat(
        [{"role": "user", "content": "test"}],
        stream=True,
        read_timeout=12,
    )

    assert result == '{"ok":true}'
    assert captured_payload["stream"] is True


def test_post_reports_non_json_http_envelope(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = "<html>gateway response</html>"
        headers = {
            "content-type": "text/html",
            "x-request-id": "request-123",
        }

        def raise_for_status(self) -> None:
            return None

        def json(self):
            raise ValueError("not JSON")

    class FakeHTTPClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(llm_module.httpx, "Client", FakeHTTPClient)

    with pytest.raises(
        LLMAPIError,
        match=r"non-JSON HTTP response.*request-123.*gateway response",
    ):
        _client().chat([{"role": "user", "content": "test"}])
