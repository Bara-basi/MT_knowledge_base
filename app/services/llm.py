from __future__ import annotations

import os
from dataclasses import dataclass
import json
import time
from typing import Any, Iterable
from urllib.parse import quote

import httpx


DEFAULT_KIMI_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_KIMI_MODEL = "kimi-k2.6"


class LLMConfigError(RuntimeError):
    """Raised when the LLM service is missing required configuration."""


class LLMAPIError(RuntimeError):
    """Raised when the remote LLM API returns an error response."""


class LLMTimeoutError(LLMAPIError):
    """Raised when the remote LLM API exceeds the configured request timeout."""


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str = DEFAULT_KIMI_BASE_URL
    model: str = DEFAULT_KIMI_MODEL
    timeout: float = 60.0
    connect_timeout: float = 10.0
    read_timeout: float = 180.0
    write_timeout: float = 60.0
    pool_timeout: float = 10.0

    @classmethod
    def from_env(cls) -> "LLMSettings":
        api_key = _get_first_env(
            "KIMI_API_KEY",
            "LLM_API_KEY",
            "OPENAI_API_KEY",
            "SILICONFLOW_API_KEY",
        )
        if not api_key:
            raise LLMConfigError(
                "Missing LLM API key. Please set KIMI_API_KEY, LLM_API_KEY, or OPENAI_API_KEY."
            )

        base_url = _get_first_env(
            "KIMI_BASE_URL",
            "KIMI_API_URL",
            "LLM_BASE_URL",
            "OPENAI_BASE_URL",
            "SILICONFLOW_BASE_URL",
            "SILICONFLOW_API_URL",
            default=DEFAULT_KIMI_BASE_URL,
        )
        model = _get_first_env(
            "KIMI_MODEL",
            "LLM_MODEL",
            "OPENAI_MODEL",
            "SILICONFLOW_MODEL",
            default=DEFAULT_KIMI_MODEL,
        )
        timeout = float(os.getenv("LLM_TIMEOUT", "180"))
        connect_timeout = float(os.getenv("LLM_CONNECT_TIMEOUT", "10"))
        read_timeout = float(os.getenv("LLM_READ_TIMEOUT", str(timeout)))
        write_timeout = float(os.getenv("LLM_WRITE_TIMEOUT", "60"))
        pool_timeout = float(os.getenv("LLM_POOL_TIMEOUT", "10"))

        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model=model,
            timeout=timeout,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
            pool_timeout=pool_timeout,
        )


def _get_first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


class LLMClient:
    """Small OpenAI-compatible chat client for KIMI and similar providers."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings.from_env()

    def chat(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        read_timeout: float | None = None,
        stream: bool = False,
    ) -> str:
        selected_model = model or self.settings.model
        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": list(messages),
        }
        normalized_temperature = _normalize_temperature(
            selected_model,
            base_url=self.settings.base_url,
            temperature=temperature,
        )
        if normalized_temperature is not None:
            payload["temperature"] = normalized_temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_body:
            payload.update(extra_body)

        if stream:
            payload["stream"] = True
            return self._stream_chat(payload, read_timeout=read_timeout)

        data = self._post("/chat/completions", payload, read_timeout=read_timeout)
        choices = data.get("choices") or []
        if not choices:
            raise LLMAPIError(f"LLM response does not contain choices: {data}")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                str(block.get("text") or "")
                for block in content
                if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
            )
        if not isinstance(content, str):
            raise LLMAPIError(f"LLM response content is not text: {data}")
        stripped = content.strip()
        if not stripped:
            finish_reason = choices[0].get("finish_reason")
            reasoning = message.get("reasoning_content")
            reasoning_length = len(reasoning) if isinstance(reasoning, str) else 0
            raise LLMAPIError(
                "LLM response content is empty "
                f"(finish_reason={finish_reason!r}, reasoning_length={reasoning_length})"
            )
        return stripped

    def _stream_chat(
        self,
        payload: dict[str, Any],
        *,
        read_timeout: float | None = None,
    ) -> str:
        """Consume an OpenAI-compatible SSE chat response incrementally."""

        url = f"{self.settings.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        chunks: list[str] = []
        try:
            with httpx.Client(
                timeout=self._build_timeout(read_timeout=read_timeout)
            ) as client:
                with client.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        event_text = line[5:].strip()
                        if event_text == "[DONE]":
                            break
                        try:
                            event = json.loads(event_text)
                        except json.JSONDecodeError as exc:
                            raise LLMAPIError(
                                f"LLM stream returned invalid JSON event: {event_text[:200]}"
                            ) from exc
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if isinstance(content, str):
                            chunks.append(content)
                        elif isinstance(content, list):
                            chunks.extend(
                                str(block.get("text") or "")
                                for block in content
                                if isinstance(block, dict)
                                and block.get("type") in {"text", "output_text"}
                            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"LLM streaming request timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMAPIError(
                f"LLM streaming API returned {exc.response.status_code}: "
                f"{exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMAPIError(f"Failed to stream LLM API: {exc}") from exc

        content = "".join(chunks).strip()
        if not content:
            raise LLMAPIError("LLM streaming response content is empty")
        return content

    def upload_file(
        self,
        *,
        filename: str,
        content: bytes,
        purpose: str,
        content_type: str = "application/octet-stream",
        read_timeout: float | None = None,
    ) -> str:
        """Upload a temporary file and return its provider file ID."""

        url = f"{self.settings.base_url}/files"
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        timeout = self._build_timeout(read_timeout=read_timeout)
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    url,
                    headers=headers,
                    data={"purpose": purpose},
                    files={"file": (filename, content, content_type)},
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"LLM file upload timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMAPIError(
                f"LLM file upload returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMAPIError(f"Failed to upload LLM file: {exc}") from exc

        payload = response.json()
        file_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(file_id, str) or not file_id.strip():
            raise LLMAPIError(f"LLM file upload response has no file ID: {payload}")
        return file_id.strip()

    def delete_file(
        self,
        file_id: str,
        *,
        read_timeout: float | None = None,
    ) -> None:
        """Delete a temporary provider file."""

        safe_file_id = quote(str(file_id), safe="")
        url = f"{self.settings.base_url}/files/{safe_file_id}"
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        timeout = self._build_timeout(read_timeout=read_timeout)
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.delete(url, headers=headers)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"LLM file deletion timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMAPIError(
                f"LLM file deletion returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMAPIError(f"Failed to delete LLM file: {exc}") from exc

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        read_timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.settings.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

        started_at = time.perf_counter()
        try:
            timeout = self._build_timeout(read_timeout=read_timeout)
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                "LLM API request timed out "
                f"(elapsed={time.perf_counter() - started_at:.2f}s, "
                f"model={payload.get('model')!r}, endpoint={url}): {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            response_text = exc.response.text
            request_id = _response_request_id(exc.response)
            raise LLMAPIError(
                f"LLM API returned {exc.response.status_code} "
                f"(elapsed={time.perf_counter() - started_at:.2f}s, "
                f"model={payload.get('model')!r}, request_id={request_id!r}): "
                f"{response_text[:2000]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMAPIError(
                "Failed to call LLM API "
                f"(elapsed={time.perf_counter() - started_at:.2f}s, "
                f"model={payload.get('model')!r}, endpoint={url}): {exc}"
            ) from exc

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMAPIError(
                "LLM API returned a non-JSON HTTP response "
                f"(status={response.status_code}, "
                f"content_type={response.headers.get('content-type')!r}, "
                f"request_id={_response_request_id(response)!r}, "
                f"body={response.text[:1000]!r})"
            ) from exc
        if not isinstance(data, dict):
            raise LLMAPIError(
                "LLM API returned an unexpected JSON envelope "
                f"(type={type(data).__name__}, request_id={_response_request_id(response)!r})"
            )
        return data

    def _build_timeout(self, *, read_timeout: float | None = None) -> httpx.Timeout:
        return httpx.Timeout(
            timeout=self.settings.timeout,
            connect=self.settings.connect_timeout,
            read=(
                self.settings.read_timeout
                if read_timeout is None
                else max(1.0, float(read_timeout))
            ),
            write=self.settings.write_timeout,
            pool=self.settings.pool_timeout,
        )


def _normalize_temperature(model: str, *, base_url: str, temperature: float) -> float | None:
    if _is_kimi_fixed_temperature_model(model, base_url=base_url):
        return None
    return temperature


def _is_kimi_fixed_temperature_model(model: str, *, base_url: str) -> bool:
    normalized_model = model.strip().lower()
    normalized_base_url = base_url.strip().lower()
    return normalized_model == "kimi-k2.6" or (
        "moonshot" in normalized_base_url and normalized_model.startswith("kimi-k2")
    )


def build_non_thinking_extra_body(*, model: str, base_url: str) -> dict[str, Any]:
    """Return the provider-specific switch for disabling model reasoning."""

    normalized_model = str(model or "").strip().lower()
    normalized_base_url = str(base_url or "").strip().lower()
    if "moonshot" in normalized_base_url and normalized_model in {
        "kimi-k2.5",
        "kimi-k2.6",
    }:
        # Moonshot's official K2.5/K2.6 API ignores the SiliconFlow-style
        # ``enable_thinking`` flag and expects this object instead.
        return {"thinking": {"type": "disabled"}}
    if "api.deepseek.com" in normalized_base_url:
        # DeepSeek's official OpenAI-compatible endpoint uses the structured
        # thinking switch. ``enable_thinking=false`` is used by gateways such
        # as SiliconFlow and may be ignored by the official endpoint, allowing
        # hidden reasoning to consume the entire output budget.
        return {"thinking": {"type": "disabled"}}
    return {"enable_thinking": False}


def _response_request_id(response: httpx.Response) -> str:
    for name in ("x-request-id", "request-id", "x-kimi-request-id", "trace-id"):
        value = response.headers.get(name)
        if value:
            return value
    return ""


def get_llm_client() -> LLMClient:
    return LLMClient()


if __name__ == "__main__":
    client = get_llm_client()
    reply = client.chat(
        [
            {"role": "system", "content": "你是企业内部知识库项目的解析助手。"},
            {"role": "user", "content": "请用一句话回复：LLM API 已接通。"},
        ],
        max_tokens=100,
    )
    print(reply)
