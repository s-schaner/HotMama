"""Four tapped corners → a serializable image↔court mapping.

The operator taps the court's baseline corners in a fixed order (near-left,
near-right, far-right, far-left, as seen on screen); everything else is
perspective math. Wide-angle lenses bend straight lines, so accuracy is best
mid-court and softens toward frame edges — the pads on every clip window and
the coarse zone model absorb that. Lens undistortion is a later refinement,
not a v1 requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .court import COURT_CORNERS, CourtPoint


class CalibrationError(ValueError):
    pass


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as err:  # pragma: no cover
        raise CalibrationError(
            "calibration needs OpenCV — install hotmama[capture]"
        ) from err
    return cv2


def _polygon_area(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


NEAR_HALF_CORNERS = (
    (0.0, 0.0),  # near-left baseline
    (9.0, 0.0),  # near-right baseline
    (9.0, 9.0),  # net-right
    (0.0, 9.0),  # net-left
)


@dataclass(frozen=True)
class CourtCalibration:
    """Image corners (pixels) of the court, plus the frame they were tapped on.

    ``mode``:
    - ``"full"`` — corners are the four baseline corners (near-L, near-R,
      far-R, far-L).
    - ``"near_half"`` — corners are near baseline then the two net-line
      corners. Far-half positions extrapolate through the same homography;
      they stay directionally useful but the far half compresses into very
      few pixels from a typical tripod, so near-half analytics are the
      trustworthy ones. Tapping net corners is far more precise than tapping
      a far baseline a few pixels tall — prefer this mode at low camera
      heights.
    """

    image_corners: tuple[tuple[float, float], ...]
    frame_width: int
    frame_height: int
    mode: str = "full"

    def __post_init__(self) -> None:
        if len(self.image_corners) != 4:
            raise CalibrationError("exactly 4 corners required")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise CalibrationError("frame size must be positive")
        if self.mode not in ("full", "near_half"):
            raise CalibrationError(f"unknown calibration mode {self.mode!r}")
        area = _polygon_area(list(self.image_corners))
        frame_area = self.frame_width * self.frame_height
        if abs(area) < frame_area * 0.01:
            raise CalibrationError("corners are collinear or the quad is too small")

    # -- transforms ---------------------------------------------------------

    def _matrices(self) -> tuple[Any, Any]:
        import numpy as np

        cv2 = _cv2()
        source = np.array(self.image_corners, dtype=np.float32)
        target_corners = COURT_CORNERS if self.mode == "full" else NEAR_HALF_CORNERS
        target = np.array(target_corners, dtype=np.float32)
        forward = cv2.getPerspectiveTransform(source, target)
        inverse = cv2.getPerspectiveTransform(target, source)
        return forward, inverse

    def image_to_court(self, points: list[tuple[float, float]]) -> list[CourtPoint]:
        import numpy as np

        cv2 = _cv2()
        forward, _ = self._matrices()
        if not points:
            return []
        array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(array, forward).reshape(-1, 2)
        return [CourtPoint(float(x), float(y)) for x, y in mapped]

    def court_to_image(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        import numpy as np

        cv2 = _cv2()
        _, inverse = self._matrices()
        if not points:
            return []
        array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(array, inverse).reshape(-1, 2)
        return [(float(x), float(y)) for x, y in mapped]

    def scaled_to(self, width: int, height: int) -> CourtCalibration:
        """The same calibration re-expressed for a different frame size."""
        sx = width / self.frame_width
        sy = height / self.frame_height
        return CourtCalibration(
            image_corners=tuple((x * sx, y * sy) for x, y in self.image_corners),
            frame_width=width,
            frame_height=height,
            mode=self.mode,
        )

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_corners": [list(corner) for corner in self.image_corners],
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CourtCalibration:
        try:
            corners = tuple(
                (float(corner[0]), float(corner[1])) for corner in data["image_corners"]
            )
            return cls(
                image_corners=corners,
                frame_width=int(data["frame_width"]),
                frame_height=int(data["frame_height"]),
                mode=str(data.get("mode", "full")),
            )
        except (KeyError, TypeError, ValueError, IndexError) as err:
            raise CalibrationError(f"invalid calibration data: {err}") from err
