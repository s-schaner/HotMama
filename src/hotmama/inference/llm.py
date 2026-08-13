"""Provider-abstracted chat clients (D4).

One OpenAI-compatible client covers the whole local fleet and most clouds
(KTransformers, LM Studio, vLLM, xAI, OpenAI, Gemini's compat endpoint);
Anthropic gets its own shape. Which endpoint actually serves a request is
purely host configuration — this repo never assumes a specific box.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import httpx

LOGGER = logging.getLogger("hotmama.inference")

_ANTHROPIC_DEFAULT_BASE = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"


class LLMError(RuntimeError):
    pass


class ChatClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str:
        """One non-streaming chat completion. Raises LLMError on failure."""
        ...


class _HttpChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        max_tokens: int,
        temperature: float,
        http: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._http = http or httpx.Client(timeout=120.0)

    def _post(self, url: str, *, headers: dict[str, str], body: dict[str, Any]) -> Any:
        try:
            response = self._http.post(url, json=body, headers=headers)
        except httpx.HTTPError as err:
            raise LLMError(f"LLM endpoint unreachable: {err}") from err
        if response.status_code != 200:
            raise LLMError(f"LLM endpoint returned {response.status_code}: {response.text[:200]}")
        return response.json()


class OpenAICompatChatClient(_HttpChatClient):
    def complete(self, system: str, user: str) -> str:
        base = self._base
        url = (
            f"{base}/chat/completions"
            if base.endswith("/v1")
            else f"{base}/v1/chat/completions"
        )
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = self._post(
            url,
            headers=headers,
            body={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            },
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as err:
            raise LLMError("malformed completion response") from err
        if not isinstance(content, str) or not content.strip():
            raise LLMError("empty completion")
        return content.strip()


class AnthropicChatClient(_HttpChatClient):
    def complete(self, system: str, user: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        data = self._post(
            f"{self._base}/v1/messages",
            headers=headers,
            body={
                "model": self.model,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        try:
            blocks = data["content"]
            text = "".join(
                block["text"] for block in blocks if block.get("type") == "text"
            )
        except (KeyError, TypeError) as err:
            raise LLMError("malformed messages response") from err
        if not text.strip():
            raise LLMError("empty completion")
        return text.strip()


def make_chat_client(
    *,
    provider: str | None,
    base_url: str | None,
    api_key: str | None,
    model: str | None,
    max_tokens: int = 500,
    temperature: float = 0.4,
    http: httpx.Client | None = None,
) -> ChatClient | None:
    """Build the configured client, or None when summaries are disabled."""
    if not provider or not model:
        return None
    provider = provider.lower()
    if provider == "openai":
        if not base_url:
            LOGGER.warning("llm_provider=openai requires llm_base_url")
            return None
        return OpenAICompatChatClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            http=http,
        )
    if provider == "anthropic":
        return AnthropicChatClient(
            base_url=base_url or _ANTHROPIC_DEFAULT_BASE,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            http=http,
        )
    LOGGER.warning("unknown llm_provider %r", provider)
    return None
