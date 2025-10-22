"""Helpers for converting natural language requests into job specs."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI
from openai import OpenAIError  # type: ignore[attr-defined]

from .config import Settings
from .schemas import JobSpec

LOGGER = logging.getLogger("hotmama.api.parser")


DEFAULT_SYSTEM_PROMPT = """
You are a strict data formatter. Convert the user's request into a valid JSON
object that exactly matches the provided JSON Schema.

Do not include any text outside JSON.

Do not invent file paths. If the user references a file by name only, set
"source_uri" to that name verbatim.

If the user omits a field, fill sensible defaults (task="analyze_video",
fps=30, priority="normal").

If time ranges are mentioned in natural language ("first 10 seconds",
"00:15 to 00:45"), normalize into "clips": [{"start":"HH:MM:SS",
"end":"HH:MM:SS"}].

If the model choice is unspecified, set "model" to "qwen-vision-default".

Never output comments, markdown, or code fences. Output JSON only.
""".strip()


class JobParserError(RuntimeError):
    """Error raised when the parser cannot fulfil a request."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class JobParser:
    """Abstract parser interface."""

    def parse(self, text: str) -> JobSpec:  # pragma: no cover - interface method
        raise NotImplementedError


@dataclass(slots=True)
class DisabledJobParser(JobParser):
    """Parser implementation used when parsing is not configured."""

    reason: str = "natural language parsing is disabled"
    status_code: int = 503

    def parse(self, text: str) -> JobSpec:
        raise JobParserError(self.reason, status_code=self.status_code)


class LMStudioJobParser(JobParser):
    """Parse natural language prompts using an LM Studio hosted model."""

    def __init__(self, settings: Settings) -> None:
        if not settings.lmstudio_base_url or not settings.lm_parser_model:
            raise ValueError("LM Studio parser requires base URL and model name")
        self._client = OpenAI(
            base_url=settings.lmstudio_base_url,
            api_key=settings.lmstudio_api_key,
        )
        self._model = settings.lm_parser_model
        self._temperature = settings.lm_parser_temperature
        self._max_tokens = settings.lm_parser_max_tokens
        self._system_prompt = settings.lm_parser_system_prompt or DEFAULT_SYSTEM_PROMPT
        self._schema = JobSpec.model_json_schema()

    def parse(self, text: str) -> JobSpec:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "job_spec",
                        "strict": True,
                        "schema": self._schema,
                    },
                },
            )
        except OpenAIError as exc:  # pragma: no cover - network failure path
            LOGGER.exception("parser request failed", extra={"error": str(exc)})
            raise JobParserError("parser request failed", status_code=502) from exc
        except Exception as exc:  # pragma: no cover - defensive catch-all
            LOGGER.exception("unexpected parser error", extra={"error": str(exc)})
            raise JobParserError("parser request failed", status_code=502) from exc

        if not response.choices:
            raise JobParserError("parser returned no choices", status_code=502)
        content = response.choices[0].message.content
        if not content:
            raise JobParserError("parser returned empty content", status_code=502)

        try:
            return JobSpec.model_validate_json(content)
        except Exception as exc:
            LOGGER.exception("invalid parser output", extra={"content": content})
            raise JobParserError("parser produced invalid output", status_code=502) from exc


def create_job_parser(settings: Settings) -> JobParser:
    """Factory that returns the appropriate parser implementation."""

    if not settings.lm_parser_model:
        return DisabledJobParser()
    try:
        return LMStudioJobParser(settings)
    except Exception as exc:  # pragma: no cover - defensive branch
        LOGGER.exception("failed to initialise LM Studio parser", exc_info=exc)
        return DisabledJobParser("unable to initialise parser")
