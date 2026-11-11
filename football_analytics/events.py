"""Ball possession and pass detection.

Possession: the ball belongs to the closest player (or goalkeeper) when it is
within ``possession_distance_ratio`` x that player's box height of the box.
Using the player's own size instead of a fixed pixel distance keeps the rule
valid at any zoom level or video resolution. A new owner must hold the ball
for ``possession_confirm_frames`` processed frames before it counts, which
filters out the ball merely flying past someone.

Possession time goes to the team of the last confirmed owner (the ball stays
"theirs" while it travels), the usual broadcast definition.

Passes: owner changes from player A to player B of the same team. If the
team changes instead, it is recorded as a turnover.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import supervision as sv

from .config import NO_TEAM, Settings


@dataclass(frozen=True)
class Owner:
    tracker_id: int
    team: int
    position: tuple[float, float]  # player's feet in image pixels


@dataclass(frozen=True)
class BallEvent:
    kind: str  # "pass" or "turnover"
    frame: int
    time_s: float
    team: int  # team that ends up with the ball
    from_player: int
    to_player: int
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]

    def to_dict(self) -> dict:
        return asdict(self)


def distance_to_boxes(point: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Euclidean distance from ``point`` to each box (0 if inside)."""
    dx = np.maximum.reduce([xyxy[:, 0] - point[0], np.zeros(len(xyxy)), point[0] - xyxy[:, 2]])
    dy = np.maximum.reduce([xyxy[:, 1] - point[1], np.zeros(len(xyxy)), point[1] - xyxy[:, 3]])
    return np.hypot(dx, dy)


def find_ball_owner(
    people: sv.Detections,
    teams: np.ndarray,
    ball: sv.Detections,
    distance_ratio: float,
) -> Owner | None:
    """The player controlling the ball in this frame, if any."""
    if len(ball) == 0 or len(people) == 0 or people.tracker_id is None:
        return None
    eligible = np.flatnonzero(teams != NO_TEAM)
    if len(eligible) == 0:
        return None
    ball_center = ball.get_anchors_coordinates(sv.Position.CENTER)[0]
    boxes = people.xyxy[eligible]
    distances = distance_to_boxes(ball_center, boxes)
    heights = boxes[:, 3] - boxes[:, 1]
    best = int(np.argmin(distances))
    if distances[best] > distance_ratio * heights[best]:
        return None
    i = eligible[best]
    feet = people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[i]
    return Owner(int(people.tracker_id[i]), int(teams[i]), (float(feet[0]), float(feet[1])))


class MatchStats:
    """Accumulates possession time, passes and turnovers over a video."""

    def __init__(self, fps: float, settings: Settings, n_teams: int = 2) -> None:
        self.fps = fps if fps > 0 else 25.0
        self.distance_ratio = settings.possession_distance_ratio
        self.confirm_frames = max(1, settings.possession_confirm_frames)
        self.n_teams = n_teams

        self.owner: Owner | None = None
        self._candidate: Owner | None = None
        self._candidate_count = 0
        self._last_owner_position: tuple[float, float] | None = None

        self.possession_frames = np.zeros(n_teams, dtype=float)
        self.events: list[BallEvent] = []
        self.timeline: list[tuple[int, float, int]] = []  # frame, time_s, team

    # ------------------------------------------------------------------ update
    def update(
        self,
        frame_index: int,
        people: sv.Detections,
        teams: np.ndarray,
        ball: sv.Detections,
        weight: float = 1.0,
    ) -> BallEvent | None:
        """Process one analysed frame. ``weight`` = video frames it stands for
        (the frame stride), so possession time stays correct when frames are
        skipped. Returns the pass/turnover completed on this frame, if any."""
        candidate = find_ball_owner(people, teams, ball, self.distance_ratio)
        event = None

        if candidate is None:
            self._candidate, self._candidate_count = None, 0
        else:
            if self._candidate and self._candidate.tracker_id == candidate.tracker_id:
                self._candidate_count += 1
            else:
                self._candidate, self._candidate_count = candidate, 1
            if self.owner and candidate.tracker_id == self.owner.tracker_id:
                self.owner = candidate
                self._last_owner_position = candidate.position
            elif self._candidate_count >= self.confirm_frames:
                event = self._change_owner(frame_index, candidate)

        if self.owner is not None:
            self.possession_frames[self.owner.team] += weight
        self.timeline.append(
            (frame_index, frame_index / self.fps, self.owner.team if self.owner else NO_TEAM)
        )
        return event

    def _change_owner(self, frame_index: int, new_owner: Owner) -> BallEvent | None:
        previous = self.owner
        self.owner = new_owner
        start = self._last_owner_position
        self._last_owner_position = new_owner.position
        if previous is None or start is None:
            return None
        kind = "pass" if previous.team == new_owner.team else "turnover"
        event = BallEvent(
            kind=kind,
            frame=frame_index,
            time_s=frame_index / self.fps,
            team=new_owner.team,
            from_player=previous.tracker_id,
            to_player=new_owner.tracker_id,
            start_xy=start,
            end_xy=new_owner.position,
        )
        self.events.append(event)
        return event

    # ----------------------------------------------------------------- results
    @property
    def possession_pct(self) -> list[float]:
        total = self.possession_frames.sum()
        if total == 0:
            return [0.0] * self.n_teams
        return [float(100 * f / total) for f in self.possession_frames]

    def count(self, kind: str, team: int) -> int:
        return sum(1 for e in self.events if e.kind == kind and e.team == team)

    @property
    def passes(self) -> list[int]:
        return [self.count("pass", t) for t in range(self.n_teams)]

    @property
    def turnovers_won(self) -> list[int]:
        return [self.count("turnover", t) for t in range(self.n_teams)]

    def summary(self) -> dict:
        return {
            "possession_pct": [round(p, 1) for p in self.possession_pct],
            "possession_seconds": [round(f / self.fps, 2) for f in self.possession_frames],
            "passes": self.passes,
            "turnovers_won": self.turnovers_won,
        }
