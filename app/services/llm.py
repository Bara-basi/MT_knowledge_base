from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import httpx


DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-V4-Flash"


class LLMConfigError(RuntimeError):
    """Raised when the LLM service is missing required configuration."""


class LLMAPIError(RuntimeError):
    """Raised when the remote LLM API returns an error response."""


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str = DEFAULT_SILICONFLOW_BASE_URL
    model: str = DEFAULT_SILICONFLOW_MODEL
    timeout: float = 60.0
    connect_timeout: float = 10.0
    read_timeout: float = 180.0
    write_timeout: float = 60.0
    pool_timeout: float = 10.0

    @classmethod
    def from_env(cls) -> "LLMSettings":
        api_key = _get_first_env(
            "SILICONFLOW_API_KEY",
            "LLM_API_KEY",
            "OPENAI_API_KEY",
        )
        if not api_key:
            raise LLMConfigError(
                "Missing LLM API key. Please set SILICONFLOW_API_KEY, LLM_API_KEY, or OPENAI_API_KEY."
            )

        base_url = _get_first_env(
            "SILICONFLOW_BASE_URL",
            "LLM_BASE_URL",
            "OPENAI_BASE_URL",
            default=DEFAULT_SILICONFLOW_BASE_URL,
        )
        model = _get_first_env(
            "SILICONFLOW_MODEL",
            "LLM_MODEL",
            "OPENAI_MODEL",
            default=DEFAULT_SILICONFLOW_MODEL,
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
    """Small OpenAI-compatible chat client for SiliconFlow and similar providers."""

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
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.settings.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_body:
            payload.update(extra_body)

        data = self._post("/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise LLMAPIError(f"LLM response does not contain choices: {data}")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMAPIError(f"LLM response content is not text: {data}")

        return content.strip()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.settings.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

        try:
            timeout = httpx.Timeout(
                timeout=self.settings.timeout,
                connect=self.settings.connect_timeout,
                read=self.settings.read_timeout,
                write=self.settings.write_timeout,
                pool=self.settings.pool_timeout,
            )
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMAPIError(f"LLM API request timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            response_text = exc.response.text
            raise LLMAPIError(f"LLM API returned {exc.response.status_code}: {response_text}") from exc
        except httpx.HTTPError as exc:
            raise LLMAPIError(f"Failed to call LLM API: {exc}") from exc

        return response.json()


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
