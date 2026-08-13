"""Provider-abstracted LLM access — endpoints are host configuration, never code."""

from .llm import (
    AnthropicChatClient,
    ChatClient,
    LLMError,
    OpenAICompatChatClient,
    make_chat_client,
)
from .summary import build_summary_facts, generate_set_summary

__all__ = [
    "AnthropicChatClient",
    "ChatClient",
    "LLMError",
    "OpenAICompatChatClient",
    "build_summary_facts",
    "generate_set_summary",
    "make_chat_client",
]
