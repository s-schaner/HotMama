"""Side-mapping and heatmap accumulation — replay-safe like everything else."""

from __future__ import annotations

import pytest

from hotmama.core import (
    CvObservation,
    EngineError,
    EventRetracted,
    SidesSet,
    apply_strict,
    replay,
)

from .helpers import base_events


def _tracks_observation(cells_value: int = 2) -> CvObservation:
    return CvObservation(
        kind="player_tracks",
        data={
            "court": {
                "cell_grid": [cells_value] * 18,
                "near_hits": 30,
                "far_hits": 6,
                "out_of_bounds_hits": 40,
            }
        },
        confidence=0.6,
    )


class TestSides:
    def test_sides_set_records_on_current_set(self) -> None:
        state = replay([*base_events(), SidesSet(our_side="far")])
        current = state.current_set
        assert current is not None
        assert current.our_side == "far"
        assert state.to_public_dict()["current_set"]["our_side"] == "far"

    def test_sides_can_flip(self) -> None:
        state = replay(
            [*base_events(), SidesSet(our_side="near"), SidesSet(our_side="far")]
        )
        current = state.current_set
        assert current is not None
        assert current.our_side == "far"

    def test_sides_requires_open_set(self) -> None:
        with pytest.raises(EngineError, match="no set in progress"):
            apply_strict(base_events()[:2], SidesSet(our_side="near"))

    def test_default_is_unset(self) -> None:
        state = replay(base_events())
        current = state.current_set
        assert current is not None
        assert current.our_side is None


class TestHeatmapAccumulation:
    def test_grids_sum_across_observations(self) -> None:
        state = replay([*base_events(), _tracks_observation(2), _tracks_observation(3)])
        assert state.heatmap_cells == [5] * 18
        assert state.heatmap_near == 60
        assert state.heatmap_far == 12
        assert state.heatmap_oob == 80
        assert state.heatmap_observations == 2

        public = state.to_public_dict()["court_heatmap"]
        assert public["cells"] == [5] * 18
        assert public["observations"] == 2

    def test_retraction_heals_heatmap(self) -> None:
        observation = _tracks_observation(4)
        state = replay(
            [
                *base_events(),
                observation,
                EventRetracted(target_event_id=observation.event_id),
            ]
        )
        assert state.heatmap_cells == [0] * 18
        assert state.heatmap_observations == 0

    def test_malformed_court_data_ignored(self) -> None:
        bad = CvObservation(kind="player_tracks", data={"court": {"cell_grid": [1, 2]}})
        worse = CvObservation(kind="player_tracks", data={"court": "nope"})
        state = replay([*base_events(), bad, worse])
        assert state.heatmap_observations == 0
        assert state.cv_observations == 2  # still counted as observations


def test_report_renders_heatmap_section() -> None:
    from hotmama.analytics import match_summary
    from hotmama.reports import render_report_html

    state = replay([*base_events(), _tracks_observation(2)])
    html = render_report_html(
        state=state.to_public_dict(),
        summary=match_summary(state),
        clips=[],
        label="Heatmap Night",
        generated_at="now",
    )
    assert "Court coverage" in html
    assert "1 analyzed rallies" in html
    assert "40 bystanders filtered" in html
