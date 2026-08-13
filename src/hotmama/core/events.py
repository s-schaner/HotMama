"""Domain events — the only way facts enter a HotMama session.

Every event is immutable and carries provenance (``producer``, ``actor``,
``confidence``). The engine never trusts a producer more than another; a CV
pipeline and a coach's thumb write to the same log. Corrections are events
too (:class:`EventRetracted`, :class:`ScoreAdjusted`) — nothing is ever
edited in place.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from .ids import new_event_id, utcnow


class Team(StrEnum):
    US = "us"
    THEM = "them"

    @property
    def other(self) -> Team:
        return Team.THEM if self is Team.US else Team.US


class SessionKind(StrEnum):
    MATCH = "match"
    PRACTICE = "practice"
    SCRIMMAGE = "scrimmage"


class Producer(StrEnum):
    HUMAN = "human"
    CV_WELL = "cv_well"
    CV_CLOUD = "cv_cloud"
    IMPORT = "import"


class PointReason(StrEnum):
    """How a rally ended.

    ``KILL``/``ACE``/``BLOCK`` are credited to the rally *winner*;
    the ``ERR_*`` codes are charged to the rally *loser*. Combined with
    ``RallyEnded.winner`` this is unambiguous without any "our/their" prefixes.
    """

    KILL = "kill"
    ACE = "ace"
    BLOCK = "block"
    ERR_SERVE = "err_serve"
    ERR_ATTACK = "err_attack"
    ERR_NET = "err_net"
    ERR_HANDLING = "err_handling"
    ERR_OTHER = "err_other"
    UNKNOWN = "unknown"


class TagCode(StrEnum):
    GREAT_SERVE = "great_serve"
    GREAT_DIG = "great_dig"
    ATTACK_WINNER = "attack_winner"
    SERVE_ERROR = "serve_error"
    DIG_FAILURE = "dig_failure"
    POSITIONING_ERROR = "positioning_error"
    HIGHLIGHT = "highlight"
    CUSTOM = "custom"


class TouchKind(StrEnum):
    SERVE = "serve"
    PASS = "pass"
    SET = "set"
    ATTACK = "attack"
    BLOCK = "block"
    DIG = "dig"
    FREEBALL = "freeball"


class Touch(BaseModel):
    """One contact in the optional per-rally touch chain (progressive entry)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: TouchKind
    team: Team = Team.US
    player_id: str | None = None
    opponent_jersey: int | None = Field(default=None, ge=0, le=99)
    grade: int | None = Field(default=None, ge=0, le=3)
    zone: int | None = Field(default=None, ge=1, le=9)


class RosterPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    player_id: str
    name: str
    jersey: int | None = Field(default=None, ge=0, le=99)
    is_libero: bool = False

    @field_validator("player_id", "name")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class BaseEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(default_factory=new_event_id)
    occurred_at: datetime = Field(default_factory=utcnow)
    producer: Producer = Producer.HUMAN
    actor: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SessionCreated(BaseEvent):
    type: Literal["session_created"] = "session_created"
    kind: SessionKind = SessionKind.MATCH
    our_team: str = "Us"
    opponent: str = "Them"
    best_of: Literal[1, 3, 5] = 5
    set_points: int = Field(default=25, ge=1, le=99)
    final_set_points: int = Field(default=15, ge=1, le=99)


class RosterRegistered(BaseEvent):
    """Declares (or replaces) our roster. Idempotent by design."""

    type: Literal["roster_registered"] = "roster_registered"
    players: list[RosterPlayer]


class SetStarted(BaseEvent):
    type: Literal["set_started"] = "set_started"
    set_number: int = Field(ge=1, le=5)
    lineup: list[str] = Field(min_length=6, max_length=6)
    """Player ids by serving order: index 0 is position 1 (first server)."""
    liberos: list[str] = Field(default_factory=list, max_length=2)
    we_serve_first: bool = True


class RallyStarted(BaseEvent):
    """Optional marker: gives clips an in-point and the UI a live-rally state."""

    type: Literal["rally_started"] = "rally_started"


class RallyEnded(BaseEvent):
    type: Literal["rally_ended"] = "rally_ended"
    winner: Team
    reason: PointReason = PointReason.UNKNOWN
    player_id: str | None = None
    """Our player most responsible (the scorer or the one charged with the error)."""
    opponent_jersey: int | None = Field(default=None, ge=0, le=99)
    touches: list[Touch] = Field(default_factory=list)


class ScoreAdjusted(BaseEvent):
    """Manual override — refs miss calls. Always available, never questioned."""

    type: Literal["score_adjusted"] = "score_adjusted"
    us_delta: int = Field(default=0, ge=-10, le=10)
    them_delta: int = Field(default=0, ge=-10, le=10)
    note: str | None = None


class MomentTagged(BaseEvent):
    type: Literal["moment_tagged"] = "moment_tagged"
    tag: TagCode
    player_id: str | None = None
    note: str | None = None
    custom_label: str | None = None


class SubMade(BaseEvent):
    """Substitution: swaps rotation identity (slot ownership) and court presence."""

    type: Literal["sub_made"] = "sub_made"
    player_in: str
    player_out: str


class LiberoSwap(BaseEvent):
    """Libero exchange: changes who is physically on court, never slot ownership."""

    type: Literal["libero_swap"] = "libero_swap"
    player_in: str
    player_out: str


class TimeoutCalled(BaseEvent):
    type: Literal["timeout_called"] = "timeout_called"
    team: Team


class SetEnded(BaseEvent):
    """Explicit human confirmation — the engine flags a decided set, people end it."""

    type: Literal["set_ended"] = "set_ended"


class SessionClosed(BaseEvent):
    type: Literal["session_closed"] = "session_closed"


class NoteAdded(BaseEvent):
    type: Literal["note_added"] = "note_added"
    text: str


class EventRetracted(BaseEvent):
    """Tombstone: the target event is treated as if it never happened."""

    type: Literal["event_retracted"] = "event_retracted"
    target_event_id: str


class CvObservation(BaseEvent):
    """Reserved for the vision phase: raw observations that may become proposals.

    The v1 engine records their count and otherwise ignores them, so Well
    workers can start emitting before the confirm-flow ships.
    """

    type: Literal["cv_observation"] = "cv_observation"
    kind: str
    data: dict[str, Any] = Field(default_factory=dict)


AnyEvent = Annotated[
    SessionCreated
    | RosterRegistered
    | SetStarted
    | RallyStarted
    | RallyEnded
    | ScoreAdjusted
    | MomentTagged
    | SubMade
    | LiberoSwap
    | TimeoutCalled
    | SetEnded
    | SessionClosed
    | NoteAdded
    | EventRetracted
    | CvObservation,
    Field(discriminator="type"),
]

EVENT_ADAPTER: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)


def parse_event(data: dict[str, Any]) -> AnyEvent:
    """Validate a raw dict (wire/store form) into a typed event."""
    return EVENT_ADAPTER.validate_python(data)


def dump_event(event: AnyEvent) -> dict[str, Any]:
    """JSON-safe dict form of an event (for the store and the wire)."""
    return cast(dict[str, Any], EVENT_ADAPTER.dump_python(event, mode="json"))
