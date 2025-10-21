"""Async LLM client with improved error handling and retry logic."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.error_handlers import LLMAPIError

logger = logging.getLogger(__name__)


class AsyncLLMClient:
    """Async client for OpenAI-compatible LLM endpoints with retry logic."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize async LLM client.

        Args:
            endpoint: Base URL for the LLM API
            model: Model identifier
            api_key: Optional API authentication token
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Send chat completion request to LLM API.

        Args:
            messages: List of message objects with role and content
            **kwargs: Additional parameters to pass to the API

        Returns:
            API response as dictionary

        Raises:
            LLMAPIError: If the API request fails
        """
        payload = {"model": self.model, "messages": messages}
        payload.update(kwargs)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info(
            "Sending LLM request",
            extra={
                "endpoint": self.endpoint,
                "model": self.model,
                "message_count": len(messages),
            },
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.endpoint}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()

                result = response.json()
                logger.info(
                    "LLM request completed",
                    extra={
                        "status_code": response.status_code,
                        "model": result.get("model"),
                    },
                )
                return result

        except httpx.TimeoutException as exc:
            logger.error("LLM request timed out", exc_info=exc)
            raise LLMAPIError(f"LLM API request timed out after {self.timeout}s")

        except httpx.HTTPStatusError as exc:
            logger.error(
                "LLM request failed with HTTP error",
                exc_info=exc,
                extra={"status_code": exc.response.status_code},
            )
            raise LLMAPIError(f"LLM API returned error: {exc.response.status_code}")

        except httpx.NetworkError as exc:
            logger.error("LLM request failed with network error", exc_info=exc)
            raise LLMAPIError("Network error connecting to LLM API")

        except Exception as exc:
            logger.exception("Unexpected error in LLM request", exc_info=exc)
            raise LLMAPIError(f"Unexpected error: {exc}")

    async def chat_with_vision(
        self,
        text: str,
        images: list[str],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Send vision-language model request with text and images.

        Args:
            text: User text prompt
            images: List of base64-encoded image data URIs
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            API response as dictionary
        """
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })

        user_content = [{"type": "text", "text": text}]
        user_content.extend([
            {"type": "image_url", "image_url": {"url": img}}
            for img in images
        ])

        messages.append({
            "role": "user",
            "content": user_content,
        })

        return await self.chat(messages, **kwargs)


async def test_connection(endpoint: str, model: str, api_key: str | None = None) -> bool:
    """
    Test connection to LLM endpoint.

    Args:
        endpoint: LLM API endpoint URL
        model: Model identifier
        api_key: Optional API key

    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = AsyncLLMClient(endpoint, model, api_key, timeout=10)
        messages = [{"role": "user", "content": "test"}]
        await client.chat(messages, max_tokens=1)
        return True
    except Exception as exc:
        logger.warning(f"LLM connection test failed: {exc}")
        return False
