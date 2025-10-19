from __future__ import annotations

import json
import re
from typing import Any, Dict

from pydantic import BaseModel, Field, ValidationError


class TimelineItem(BaseModel):
    t: float
    event: str
    actor: str | None = None


class RallySchema(BaseModel):
    rally_state: str
    who_won: str | None = None
    reason: str | None = None
    timeline: list[TimelineItem] = Field(default_factory=list)


JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def parse_rally(text: str) -> RallySchema:
    candidates = [text]
    match = JSON_PATTERN.search(text)
    if match:
        candidates.append(match.group(0))
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        try:
            return RallySchema(**payload)
        except ValidationError as exc:
            last_error = exc
            continue
    raise ValueError(f"Unable to parse rally response: {last_error}")
