"""Multi-object tracking.

People (players, goalkeepers, referees) are tracked with ByteTrack from
Roboflow's ``trackers`` package, which gives each person a stable id across
frames. The ball is not tracked by id; instead ``BallTracker`` keeps a short
history of positions and picks the candidate closest to it, which removes
most false positives (heads, socks, penalty spots) and fills one-frame gaps.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker


class PeopleTracker:
    """Assigns persistent ``tracker_id``s to people detections."""

    def __init__(self, frame_rate: float, lost_track_seconds: float = 1.0) -> None:
        self._frame_rate = max(frame_rate, 1.0)
        self._lost_track_seconds = lost_track_seconds
        self.reset()

    def reset(self) -> None:
        self._tracker = ByteTrackTracker(
            frame_rate=self._frame_rate,
            lost_track_buffer=max(1, round(self._frame_rate * self._lost_track_seconds)),
            track_activation_threshold=0.4,
            high_conf_det_threshold=0.5,
            minimum_consecutive_frames=1,
        )

    def update(self, people: sv.Detections) -> sv.Detections:
        """Return the tracked people; unconfirmed detections are dropped."""
        tracked = self._tracker.update(people)
        if len(tracked) == 0 or tracked.tracker_id is None:
            return sv.Detections.empty()
        return tracked[tracked.tracker_id >= 0]


class BallTracker:
    """Chooses the most plausible ball among candidates, frame by frame.

    Args:
        buffer_size: How many recent positions to average as the reference.
        max_missing: How many consecutive frames the last position is kept
            (and reported) when the ball is not detected, e.g. hidden by a
            player's leg.
        max_jump_ratio: Candidates further than this fraction of the frame
            diagonal from the reference are rejected as outliers, unless the
            ball has been lost for longer than ``max_missing``.
    """

    def __init__(
        self,
        frame_size: tuple[int, int],
        buffer_size: int = 10,
        max_missing: int = 5,
        max_jump_ratio: float = 0.25,
    ) -> None:
        width, height = frame_size
        self._max_jump = max_jump_ratio * float(np.hypot(width, height))
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._max_missing = max_missing
        self._missing = 0
        self._last: sv.Detections = sv.Detections.empty()

    def reset(self) -> None:
        self._buffer.clear()
        self._missing = 0
        self._last = sv.Detections.empty()

    def update(self, candidates: sv.Detections) -> sv.Detections:
        """Return at most one ball detection for the current frame."""
        if len(candidates) > 0:
            centers = candidates.get_anchors_coordinates(sv.Position.CENTER)
            if self._buffer and self._missing <= self._max_missing:
                reference = np.mean(self._buffer, axis=0)
                distances = np.linalg.norm(centers - reference, axis=1)
                best = int(np.argmin(distances))
                accept = distances[best] <= self._max_jump
            else:
                best = int(np.argmax(candidates.confidence))
                accept = True
            if accept:
                self._buffer.append(centers[best])
                self._missing = 0
                self._last = candidates[[best]]
                return self._last

        self._missing += 1
        if self._missing <= self._max_missing and len(self._last) > 0:
            return self._last
        self._buffer.clear()
        return sv.Detections.empty()
