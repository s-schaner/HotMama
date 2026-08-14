"""Chat clients against mock transports — the salvaged MockTransport idiom."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from hotmama.inference import (
    LLMError,
    build_summary_facts,
    generate_set_summary,
    make_chat_client,
)


def _openai_client(handler: Any, base_url: str = "http://well:8080") -> Any:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return make_chat_client(
        provider="openai",
        base_url=base_url,
        api_key="secret",
        model="qwen3-30b",
        http=http,
    )


class TestOpenAICompat:
    def test_request_shape_and_response(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": " Serve tough. "}}]},
            )

        client = _openai_client(handler)
        assert client is not None
        result = client.complete("sys prompt", "user prompt")

        assert result == "Serve tough."
        assert seen["url"] == "http://well:8080/v1/chat/completions"
        assert seen["auth"] == "Bearer secret"
        assert seen["body"]["model"] == "qwen3-30b"
        assert seen["body"]["messages"][0] == {"role": "system", "content": "sys prompt"}

    def test_base_url_already_versioned(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "http://well:8080/v1/chat/completions"
            return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

        client = _openai_client(handler, base_url="http://well:8080/v1")
        assert client is not None
        assert client.complete("s", "u") == "x"

    def test_http_error_becomes_llm_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        client = _openai_client(handler)
        assert client is not None
        with pytest.raises(LLMError, match="500"):
            client.complete("s", "u")

    def test_empty_content_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]})

        client = _openai_client(handler)
        assert client is not None
        with pytest.raises(LLMError, match="empty"):
            client.complete("s", "u")


class TestAnthropic:
    def test_request_shape_and_response(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["key"] = request.headers.get("x-api-key")
            seen["version"] = request.headers.get("anthropic-version")
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "Attack line."}]},
            )

        http = httpx.Client(transport=httpx.MockTransport(handler))
        client = make_chat_client(
            provider="anthropic",
            base_url=None,
            api_key="ak",
            model="claude-sonnet-5",
            http=http,
        )
        assert client is not None
        assert client.complete("sys", "usr") == "Attack line."
        assert seen["url"] == "https://api.anthropic.com/v1/messages"
        assert seen["key"] == "ak"
        assert seen["version"] is not None
        assert seen["body"]["system"] == "sys"
        assert seen["body"]["messages"] == [{"role": "user", "content": "usr"}]


class TestFactory:
    def test_unconfigured_returns_none(self) -> None:
        assert make_chat_client(provider=None, base_url=None, api_key=None, model=None) is None
        assert (
            make_chat_client(provider="openai", base_url=None, api_key=None, model="m")
            is None
        )
        assert (
            make_chat_client(provider="alien", base_url="http://x", api_key=None, model="m")
            is None
        )


class TestSummaryBuilder:
    def test_facts_are_compact_and_grounded(self) -> None:
        from hotmama.analytics import match_summary
        from hotmama.core import replay
        from tests.analytics.test_projections import scenario

        state = replay(scenario())
        facts = build_summary_facts(state.to_public_dict(), match_summary(state))
        assert facts["our_team"] == "HotMama"
        assert facts["current_set_score"] == "4-7"
        assert facts["biggest_leak"] and "Rotation 3" in facts["biggest_leak"]
        rotations = {row["rotation"]: row for row in facts["rotations"]}
        assert rotations[3]["lost_reasons"] == {"err_attack": 3}
        # Only players with actual actions make the prompt.
        assert all(
            line["kills"] or line["aces"] or line["blocks"] or line["total_errors"]
            for line in facts["players"]
        )

    def test_generate_uses_client(self) -> None:
        class FakeChat:
            model = "fake"

            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            def complete(self, system: str, user: str) -> str:
                self.calls.append((system, user))
                return "three paragraphs"

        from hotmama.analytics import match_summary
        from hotmama.core import replay
        from tests.analytics.test_projections import scenario

        state = replay(scenario())
        fake = FakeChat()
        text = generate_set_summary(fake, state.to_public_dict(), match_summary(state))
        assert text == "three paragraphs"
        system, user = fake.calls[0]
        assert "volleyball" in system.lower()
        assert "Rotation 3" in user
