from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

import requests

LOGGER = logging.getLogger(__name__)


class LLMClient:
    """Minimal client supporting OpenAI-compatible endpoints."""

    def __init__(self, endpoint: str, model: str, api_key: Optional[str] = None) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key

    def chat(self, messages: Iterable[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        payload = {"model": self.model, "messages": list(messages)}
        payload.update(kwargs)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(f"{self.endpoint}/v1/chat/completions", json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
