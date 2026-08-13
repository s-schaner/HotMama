"""Identifier and clock helpers for the core domain."""

from __future__ import annotations

import secrets
import time
from datetime import UTC, datetime


def new_event_id() -> str:
    """Time-prefixed random id.

    Sortable enough to eyeball in a debugger; the authoritative ordering of
    events is always the store-assigned ``seq``, never the id.
    """
    return f"{int(time.time() * 1000):013x}{secrets.token_hex(5)}"


def new_session_id() -> str:
    return f"s_{secrets.token_hex(8)}"


def utcnow() -> datetime:
    return datetime.now(UTC)
