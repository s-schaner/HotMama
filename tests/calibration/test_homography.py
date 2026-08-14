"""Calibration math against known geometry and the real-footage scenario."""

from __future__ import annotations

import pytest

pytest.importorskip("cv2")

from hotmama.calibration import (  # noqa: E402
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    CalibrationError,
    CourtCalibration,
    CourtPoint,
)

# A frame where the full court fills a clean rectangle: y=1000 is the near
# baseline, y=0 the far one — so image y maps inversely to court depth.
RECT = CourtCalibration(
    image_corners=((0.0, 1000.0), (900.0, 1000.0), (900.0, 0.0), (0.0, 0.0)),
    frame_width=900,
    frame_height=1000,
)


class TestFullCourt:
    def test_corners_map_exactly(self) -> None:
        points = RECT.image_to_court([(0, 1000), (900, 1000), (900, 0), (0, 0)])
        assert [(round(p.x, 3), round(p.y, 3)) for p in points] == [
            (0.0, 0.0),
            (COURT_WIDTH_M, 0.0),
            (COURT_WIDTH_M, COURT_LENGTH_M),
            (0.0, COURT_LENGTH_M),
        ]

    def test_midpoint_and_roundtrip(self) -> None:
        center = RECT.image_to_court([(450, 500)])[0]
        assert center.x == pytest.approx(4.5, abs=0.01)
        assert center.y == pytest.approx(9.0, abs=0.01)
        back = RECT.court_to_image([(center.x, center.y)])[0]
        assert back[0] == pytest.approx(450, abs=0.5)
        assert back[1] == pytest.approx(500, abs=0.5)

    def test_halves(self) -> None:
        assert RECT.image_to_court([(450, 800)])[0].half == "near"
        assert RECT.image_to_court([(450, 200)])[0].half == "far"


class TestNearHalfMode:
    def test_far_positions_extrapolate(self) -> None:
        # Corners cover only the near half: near baseline at y=1000, net at y=500.
        cal = CourtCalibration(
            image_corners=((0.0, 1000.0), (900.0, 1000.0), (900.0, 500.0), (0.0, 500.0)),
            frame_width=900,
            frame_height=1000,
            mode="near_half",
        )
        net = cal.image_to_court([(450, 500)])[0]
        assert net.y == pytest.approx(9.0, abs=0.01)
        beyond = cal.image_to_court([(450, 250)])[0]
        assert beyond.half == "far"
        assert beyond.y > 9.0


class TestZones:
    def test_zone_layout(self) -> None:
        # Near half: right-back is zone 1, front-left zone 4.
        assert CourtPoint(8.0, 1.0).zone == 1
        assert CourtPoint(4.5, 1.0).zone == 6
        assert CourtPoint(1.0, 1.0).zone == 5
        assert CourtPoint(8.0, 8.0).zone == 2
        assert CourtPoint(4.5, 8.0).zone == 3
        assert CourtPoint(1.0, 8.0).zone == 4
        # Far half mirrors (their zone 1 is camera-left back).
        assert CourtPoint(1.0, 17.0).zone == 1
        assert CourtPoint(8.0, 17.0).zone == 5
        assert CourtPoint(-1.0, 5.0).zone is None

    def test_out_of_bounds(self) -> None:
        assert not CourtPoint(9.5, 4.0).in_bounds
        assert not CourtPoint(4.0, 18.5).in_bounds
        assert CourtPoint(0.0, 0.0).in_bounds


class TestSerializationAndValidation:
    def test_round_trip_with_mode(self) -> None:
        cal = CourtCalibration(
            image_corners=((10.0, 90.0), (200.0, 92.0), (180.0, 20.0), (30.0, 22.0)),
            frame_width=220,
            frame_height=100,
            mode="near_half",
        )
        restored = CourtCalibration.from_dict(cal.to_dict())
        assert restored == cal
        assert restored.mode == "near_half"

    def test_scaled_to(self) -> None:
        doubled = RECT.scaled_to(1800, 2000)
        point = doubled.image_to_court([(900, 1000)])[0]
        assert point.x == pytest.approx(4.5, abs=0.01)
        assert point.y == pytest.approx(9.0, abs=0.01)

    def test_scaled_to_preserves_mode(self) -> None:
        # Regression: mode was dropped on rescale, silently remapping
        # near-half calibrations as full-court (caught on real footage).
        near = CourtCalibration(
            image_corners=RECT.image_corners,
            frame_width=900,
            frame_height=1000,
            mode="near_half",
        )
        assert near.scaled_to(1800, 2000).mode == "near_half"

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(CalibrationError, match="4 corners"):
            CourtCalibration(
                image_corners=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
                frame_width=100,
                frame_height=100,
            )
        with pytest.raises(CalibrationError, match="collinear|small"):
            CourtCalibration(
                image_corners=((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)),
                frame_width=100,
                frame_height=100,
            )
        with pytest.raises(CalibrationError, match="mode"):
            CourtCalibration(
                image_corners=((0.0, 100.0), (100.0, 100.0), (100.0, 0.0), (0.0, 0.0)),
                frame_width=100,
                frame_height=100,
                mode="sideways",
            )
        with pytest.raises(CalibrationError, match="invalid"):
            CourtCalibration.from_dict({"image_corners": "nope"})
