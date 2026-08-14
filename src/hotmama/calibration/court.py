"""The court coordinate system every spatial feature shares.

Official indoor court: 18m end to end, 9m wide. Coordinates are meters in
"court space": x ∈ [0, 9] left→right and y ∈ [0, 18] near→far **as seen from
the calibrated camera**; the net sits at y = 9, attack lines at y = 6 and 12.
"near"/"far" are camera-relative — mapping them to us/them is a session
concern (teams switch sides), never baked into calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

COURT_WIDTH_M = 9.0
COURT_LENGTH_M = 18.0
NET_Y = 9.0
NEAR_ATTACK_Y = 6.0
FAR_ATTACK_Y = 12.0


@dataclass(frozen=True)
class CourtPoint:
    x: float
    y: float

    @property
    def in_bounds(self) -> bool:
        return 0.0 <= self.x <= COURT_WIDTH_M and 0.0 <= self.y <= COURT_LENGTH_M

    @property
    def half(self) -> str:
        """"near" | "far" — camera-relative half (net line belongs to neither)."""
        if self.y < NET_Y:
            return "near"
        if self.y > NET_Y:
            return "far"
        return "net"

    @property
    def zone(self) -> int | None:
        """Conventional 1-6 zone within the point's half, or None off-court.

        Zones are laid out serving-order style per half, from that half's own
        baseline: 1 = back-right, 6 = back-center, 5 = back-left,
        2 = front-right, 3 = front-center, 4 = front-left — mirrored so each
        team's zone 1 is its serving corner.
        """
        if not self.in_bounds:
            return None
        if self.y < NET_Y:
            depth_back = self.y < NEAR_ATTACK_Y  # back row of the near half
            thirds = min(2, int(self.x / (COURT_WIDTH_M / 3)))
            right, center, left = 2, 1, 0
        else:
            depth_back = self.y > FAR_ATTACK_Y
            thirds = min(2, int(self.x / (COURT_WIDTH_M / 3)))
            # Far half is rotated 180° relative to the camera.
            right, center, left = 0, 1, 2
        if depth_back:
            return {right: 1, center: 6, left: 5}[thirds]
        return {right: 2, center: 3, left: 4}[thirds]


COURT_CORNERS = (
    (0.0, 0.0),  # near-left
    (COURT_WIDTH_M, 0.0),  # near-right
    (COURT_WIDTH_M, COURT_LENGTH_M),  # far-right
    (0.0, COURT_LENGTH_M),  # far-left
)

# The lines a calibration overlay draws, as (start, end) in court meters.
COURT_LINES: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = (
    ((0.0, 0.0), (COURT_WIDTH_M, 0.0)),  # near baseline
    ((0.0, COURT_LENGTH_M), (COURT_WIDTH_M, COURT_LENGTH_M)),  # far baseline
    ((0.0, 0.0), (0.0, COURT_LENGTH_M)),  # left sideline
    ((COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M)),  # right sideline
    ((0.0, NET_Y), (COURT_WIDTH_M, NET_Y)),  # net line
    ((0.0, NEAR_ATTACK_Y), (COURT_WIDTH_M, NEAR_ATTACK_Y)),  # near attack line
    ((0.0, FAR_ATTACK_Y), (COURT_WIDTH_M, FAR_ATTACK_Y)),  # far attack line
)
