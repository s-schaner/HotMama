"""Event wire-format guarantees: strict schemas, stable round-trips."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hotmama.core import (
    PointReason,
    RallyEnded,
    Team,
    Touch,
    TouchKind,
    dump_event,
    parse_event,
)


def test_round_trip_preserves_event() -> None:
    original = RallyEnded(
        winner=Team.US,
        reason=PointReason.KILL,
        player_id="p4",
        opponent_jersey=7,
        touches=[Touch(kind=TouchKind.ATTACK, player_id="p4", grade=3, zone=4)],
    )
    restored = parse_event(dump_event(original))
    assert restored == original


def test_dump_is_json_safe() -> None:
    payload = dump_event(RallyEnded(winner=Team.THEM))
    assert payload["type"] == "rally_ended"
    assert payload["winner"] == "them"
    assert isinstance(payload["occurred_at"], str)


def test_unknown_type_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_event({"type": "made_up_event"})


def test_extra_fields_rejected() -> None:
    data = dump_event(RallyEnded(winner=Team.US))
    data["sneaky"] = "field"
    with pytest.raises(ValidationError):
        parse_event(data)


def test_confidence_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        RallyEnded(winner=Team.US, confidence=1.5)


def test_missing_discriminator_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_event({"winner": "us"})
