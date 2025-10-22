from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, validator


class SessionCreate(BaseModel):
    title: str
    venue: Optional[str] = None
    meta: dict = Field(default_factory=dict)


class SessionOut(BaseModel):
    id: str
    created_utc: str
    title: str
    venue: Optional[str] = None
    meta: dict


class TeamSet(BaseModel):
    session_id: str
    side: Literal["A", "B"]
    name: Optional[str]


class RosterEntry(BaseModel):
    number: str
    player_name: str


class ClipCreate(BaseModel):
    path: str
    duration_sec: float
    meta: dict = Field(default_factory=dict)


class EventIn(BaseModel):
    t_sec: float
    event_type: str
    actor_team: Optional[str] = None
    actor_number: Optional[str] = None
    actor_resolved_name: Optional[str] = None
    payload: dict = Field(default_factory=dict)


class RollupEntry(BaseModel):
    session_id: str
    team_side: Optional[str]
    number: Optional[str]
    event_type: str
    count: int


class ClipOut(BaseModel):
    id: str
    session_id: str
    path: str
    duration_sec: float
    meta: dict


class EventOut(EventIn):
    id: str
    session_id: str
    clip_id: str


class ExportResult(BaseModel):
    csv_paths: dict[str, Path]


@validator("duration_sec", pre=True)
def _duration_positive(cls, value: float) -> float:
    if value <= 0:
        raise ValueError("Clip duration must be positive")
    return value
