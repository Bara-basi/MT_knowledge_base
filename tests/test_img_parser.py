from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.llm import LLMAPIError, LLMTimeoutError
from app.services.parser import img_parser
from app.services.parser.img_parser import _analyze_image, request_multimodal_text


def test_moonshot_multimodal_request_uses_file_reference_and_cleans_up() -> None:
    class FakeClient:
        settings = SimpleNamespace(base_url="https://api.moonshot.cn/v1")

        def __init__(self) -> None:
            self.uploaded = 0
            self.deleted: list[str] = []
            self.extra_body = None

        def upload_file(self, **kwargs) -> str:
            assert kwargs["content"] == b"jpeg"
            self.uploaded += 1
            return "file-test"

        def chat(self, messages, **kwargs) -> str:
            assert messages[0]["content"][1]["image_url"]["url"] == "ms://file-test"
            assert kwargs["stream"] is False
            self.extra_body = kwargs["extra_body"]
            return "OK"

        def delete_file(self, file_id: str, **_kwargs) -> None:
            self.deleted.append(file_id)

    client = FakeClient()

    assert request_multimodal_text(
        client,
        prompt="diagnostic",
        image_bytes=b"jpeg",
    ) == "OK"
    assert client.uploaded == 1
    assert client.deleted == ["file-test"]
    assert client.extra_body == {"thinking": {"type": "disabled"}}


def test_image_timeout_is_not_retried_without_json_mode(tmp_path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"jpeg")

    class TimeoutClient:
        settings = SimpleNamespace(base_url="https://example.invalid/v1")

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, *_args, **_kwargs) -> str:
            self.calls += 1
            raise LLMTimeoutError("read timed out")

    client = TimeoutClient()
    context = {
        "document_title": "test",
        "heading_path": "",
        "nearest_body_text": "",
    }

    with pytest.raises(LLMTimeoutError):
        _analyze_image(
            {"path": str(image_path)},
            context,
            client=client,
            model="test-model",
            group_position=1,
            group_size=1,
        )

    assert client.calls == 1


def test_moonshot_upload_retries_connection_reset_then_succeeds(monkeypatch) -> None:
    class FlakyUploadClient:
        settings = SimpleNamespace(base_url="https://api.moonshot.cn/v1")

        def __init__(self) -> None:
            self.upload_calls = 0
            self.deleted: list[str] = []

        def upload_file(self, **_kwargs) -> str:
            self.upload_calls += 1
            if self.upload_calls == 1:
                raise LLMAPIError(
                    "Failed to upload LLM file: [WinError 10054] "
                    "远程主机强迫关闭了一个现有的连接"
                )
            return "file-after-retry"

        def chat(self, *_args, **_kwargs) -> str:
            return "OK"

        def delete_file(self, file_id: str, **_kwargs) -> None:
            self.deleted.append(file_id)

    monkeypatch.setattr(img_parser, "VISION_UPLOAD_RETRY_BASE_SECONDS", 0)
    client = FlakyUploadClient()

    assert request_multimodal_text(
        client,
        prompt="retry upload",
        image_bytes=b"jpeg",
    ) == "OK"
    assert client.upload_calls == 2
    assert client.deleted == ["file-after-retry"]


def test_moonshot_upload_does_not_retry_nontransient_400(monkeypatch) -> None:
    class InvalidUploadClient:
        settings = SimpleNamespace(base_url="https://api.moonshot.cn/v1")

        def __init__(self) -> None:
            self.upload_calls = 0

        def upload_file(self, **_kwargs) -> str:
            self.upload_calls += 1
            raise LLMAPIError("LLM file upload returned 400: invalid purpose")

    monkeypatch.setattr(img_parser, "VISION_UPLOAD_RETRY_BASE_SECONDS", 0)
    client = InvalidUploadClient()

    with pytest.raises(LLMAPIError, match="returned 400"):
        request_multimodal_text(
            client,
            prompt="invalid upload",
            image_bytes=b"jpeg",
        )
    assert client.upload_calls == 1


def test_moonshot_upload_exhausts_transient_retries(monkeypatch) -> None:
    class ResetUploadClient:
        settings = SimpleNamespace(base_url="https://api.moonshot.cn/v1")

        def __init__(self) -> None:
            self.upload_calls = 0

        def upload_file(self, **_kwargs) -> str:
            self.upload_calls += 1
            raise LLMAPIError(
                "Failed to upload LLM file: [WinError 10054] "
                "远程主机强迫关闭了一个现有的连接"
            )

    monkeypatch.setattr(img_parser, "VISION_UPLOAD_RETRY_BASE_SECONDS", 0)
    monkeypatch.setattr(img_parser, "VISION_UPLOAD_MAX_ATTEMPTS", 3)
    client = ResetUploadClient()

    with pytest.raises(LLMAPIError, match="WinError 10054"):
        request_multimodal_text(
            client,
            prompt="persistent reset",
            image_bytes=b"jpeg",
        )
    assert client.upload_calls == 3
