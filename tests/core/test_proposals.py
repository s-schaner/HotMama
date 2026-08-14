"""Confirm-flow at the engine level: proposals pend, resolve, and dismiss."""

from __future__ import annotations

from hotmama.core import (
    CvObservation,
    EventRetracted,
    RallyEnded,
    Team,
    replay,
)

from .helpers import base_events


def _observation(confidence: float = 0.6) -> CvObservation:
    return CvObservation(
        kind="rally_end_detected",
        data={"clip_id": "c_x"},
        confidence=confidence,
        proposal={"type": "rally_ended", "winner": "us", "reason": "kill"},
    )


def test_observation_with_proposal_pends() -> None:
    observation = _observation()
    state = replay([*base_events(), observation])
    assert state.cv_observations == 1
    assert list(state.proposals) == [observation.event_id]
    record = state.proposals[observation.event_id]
    assert record.proposal["winner"] == "us"
    assert record.confidence == 0.6

    public = state.to_public_dict()
    assert public["proposals"][0]["kind"] == "rally_end_detected"


def test_observation_without_proposal_does_not_pend() -> None:
    observation = CvObservation(kind="clip_stats", data={})
    state = replay([*base_events(), observation])
    assert state.cv_observations == 1
    assert state.proposals == {}


def test_confirming_event_resolves_proposal() -> None:
    observation = _observation()
    confirmed = RallyEnded(
        winner=Team.US, source_event_id=observation.event_id, actor="coach"
    )
    state = replay([*base_events(), observation, confirmed])
    assert state.proposals == {}
    current = state.current_set
    assert current is not None
    assert current.us_points == 1


def test_dismissal_via_retraction_removes_proposal() -> None:
    observation = _observation()
    state = replay(
        [
            *base_events(),
            observation,
            EventRetracted(target_event_id=observation.event_id),
        ]
    )
    assert state.proposals == {}
    assert state.cv_observations == 0  # retracted observations never applied


def test_unrelated_source_link_is_harmless() -> None:
    state = replay(
        [*base_events(), RallyEnded(winner=Team.US, source_event_id="nonexistent")]
    )
    current = state.current_set
    assert current is not None
    assert current.us_points == 1
    assert state.proposals == {}
