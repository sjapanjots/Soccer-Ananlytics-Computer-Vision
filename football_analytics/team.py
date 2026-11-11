"""Automatic team assignment from kit colour - no hardcoded team colours.

1. For each player box, crop the torso and drop grass-green pixels.
2. Describe the shirt by the median colour of what is left (CIE Lab space).
3. Fit K-Means with k=2 on shirts sampled across the video.
4. Each tracked player keeps a rolling majority vote of predictions, so the
   label does not flicker and is only recomputed now and then (cheap).
5. Goalkeepers wear different kits, so they join the team whose outfield
   players are, on average, closest to them.
"""

from __future__ import annotations

from collections import Counter, deque

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans

from .config import GOALKEEPER_CLASS_ID, NO_TEAM, PLAYER_CLASS_ID

# Lightness varies a lot with shadows / stadium lights; weight it down so
# hue and saturation drive the clustering.
_L_WEIGHT = 0.5

# OpenCV HSV ranges for pitch grass (H is 0-179).
_GREEN_LOW = np.array([35, 40, 40], dtype=np.uint8)
_GREEN_HIGH = np.array([85, 255, 255], dtype=np.uint8)


def torso_crop(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Crop the shirt area: upper-middle part of the player's box."""
    x1, y1, x2, y2 = xyxy.astype(float)
    w, h = x2 - x1, y2 - y1
    top, bottom = int(y1 + 0.15 * h), int(y1 + 0.5 * h)
    left, right = int(x1 + 0.2 * w), int(x2 - 0.2 * w)
    height, width = frame.shape[:2]
    top, bottom = max(top, 0), min(bottom, height)
    left, right = max(left, 0), min(right, width)
    if bottom <= top or right <= left:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[top:bottom, left:right]


def kit_feature(crop: np.ndarray) -> np.ndarray | None:
    """Return a 3-d Lab colour describing the shirt, or None if unusable."""
    if crop.size == 0 or min(crop.shape[:2]) < 2:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    not_green = cv2.inRange(hsv, _GREEN_LOW, _GREEN_HIGH) == 0
    pixels = crop[not_green]
    if len(pixels) < 0.1 * crop.shape[0] * crop.shape[1]:
        pixels = crop.reshape(-1, 3)
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    feature = np.median(lab, axis=0).astype(np.float32)
    feature[0] *= _L_WEIGHT
    return feature


def extract_features(
    frame: np.ndarray, detections: sv.Detections
) -> tuple[np.ndarray, np.ndarray]:
    """Kit features for each detection.

    Returns ``(features, valid_index)`` where ``valid_index`` holds the
    positions in ``detections`` that produced a usable feature.
    """
    features, valid = [], []
    for i, xyxy in enumerate(detections.xyxy):
        feature = kit_feature(torso_crop(frame, xyxy))
        if feature is not None:
            features.append(feature)
            valid.append(i)
    if not features:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=int)
    return np.stack(features), np.array(valid, dtype=int)


def _lab_feature_to_bgr(feature: np.ndarray) -> tuple[int, int, int]:
    lab = feature.astype(np.float32).copy()
    lab[0] /= _L_WEIGHT
    lab = np.clip(lab, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    b, g, r = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).reshape(3)
    return int(b), int(g), int(r)


class TeamClassifier:
    """K-Means (k=2) over shirt colours."""

    def __init__(self, n_teams: int = 2, random_state: int = 0) -> None:
        self.n_teams = n_teams
        self._kmeans = KMeans(n_clusters=n_teams, n_init=10, random_state=random_state)
        self.fitted = False

    def fit(self, features: np.ndarray) -> "TeamClassifier":
        if len(features) < self.n_teams:
            raise ValueError(
                f"need at least {self.n_teams} player crops to fit teams, got {len(features)}"
            )
        self._kmeans.fit(features)
        # Stable order: the lighter kit is always team 0.
        order = np.argsort(-self._kmeans.cluster_centers_[:, 0])
        self._remap = np.empty_like(order)
        self._remap[order] = np.arange(self.n_teams)
        self.centers = self._kmeans.cluster_centers_[order]
        self.fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TeamClassifier.fit must be called first")
        if len(features) == 0:
            return np.empty(0, dtype=int)
        return self._remap[self._kmeans.predict(features)]

    @property
    def team_colors_bgr(self) -> list[tuple[int, int, int]]:
        """The average kit colour of each team, usable for drawing."""
        if not self.fitted:
            raise RuntimeError("TeamClassifier.fit must be called first")
        return [_lab_feature_to_bgr(center) for center in self.centers]


class TeamAssigner:
    """Gives every tracked player / goalkeeper a stable team id.

    If the classifier has not been fitted (e.g. calibration found too few
    players), features are collected during processing and the classifier is
    fitted as soon as ``min_samples`` crops are available.
    """

    def __init__(
        self,
        classifier: TeamClassifier | None = None,
        vote_window: int = 15,
        confirm_votes: int = 5,
        refresh_every: int = 10,
        min_samples: int = 30,
    ) -> None:
        self.classifier = classifier or TeamClassifier()
        self.vote_window = vote_window
        self.confirm_votes = confirm_votes
        self.refresh_every = refresh_every
        self.min_samples = min_samples
        self._votes: dict[int, deque[int]] = {}
        self._sightings: Counter[int] = Counter()
        self._pending: list[np.ndarray] = []

    @property
    def ready(self) -> bool:
        return self.classifier.fitted

    def calibrate(self, features: np.ndarray) -> bool:
        """Fit the classifier from features sampled across the video."""
        if len(features) < max(self.classifier.n_teams, 4):
            self._pending.extend(features)
            return False
        self.classifier.fit(features)
        return True

    def _vote(self, tracker_id: int, team: int) -> None:
        votes = self._votes.setdefault(tracker_id, deque(maxlen=self.vote_window))
        votes.append(int(team))

    def _team_of(self, tracker_id: int) -> int:
        votes = self._votes.get(tracker_id)
        if not votes:
            return NO_TEAM
        return Counter(votes).most_common(1)[0][0]

    def assign(self, frame: np.ndarray, people: sv.Detections) -> np.ndarray:
        """Team id per detection in ``people`` (NO_TEAM for referees)."""
        teams = np.full(len(people), NO_TEAM, dtype=int)
        if len(people) == 0 or people.tracker_id is None:
            return teams

        players = np.flatnonzero(people.class_id == PLAYER_CLASS_ID)
        tracker_ids = people.tracker_id

        if not self.ready:
            features, _ = extract_features(frame, people[players])
            self._pending.extend(features)
            if len(self._pending) >= self.min_samples:
                self.classifier.fit(np.stack(self._pending))
                self._pending.clear()
            else:
                return teams

        # Only (re)classify players that are new or due for a refresh.
        due = []
        for i in players:
            tid = int(tracker_ids[i])
            seen = self._sightings[tid]
            self._sightings[tid] += 1
            if seen < self.confirm_votes or seen % self.refresh_every == 0:
                due.append(i)
        if due:
            due = np.array(due, dtype=int)
            features, valid = extract_features(frame, people[due])
            for idx, team in zip(due[valid], self.classifier.predict(features)):
                self._vote(int(tracker_ids[idx]), int(team))

        for i in players:
            teams[i] = self._team_of(int(tracker_ids[i]))

        self._assign_goalkeepers(people, teams)
        return teams

    def _assign_goalkeepers(self, people: sv.Detections, teams: np.ndarray) -> None:
        goalkeepers = np.flatnonzero(people.class_id == GOALKEEPER_CLASS_ID)
        if len(goalkeepers) == 0:
            return
        feet = people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        centroids = {}
        for team in range(self.classifier.n_teams):
            members = feet[teams == team]
            if len(members):
                centroids[team] = members.mean(axis=0)
        if len(centroids) < self.classifier.n_teams:
            # Can't decide this frame; fall back to the goalkeeper's history.
            for i in goalkeepers:
                teams[i] = self._team_of(int(people.tracker_id[i]))
            return
        for i in goalkeepers:
            team = min(centroids, key=lambda t: np.linalg.norm(feet[i] - centroids[t]))
            self._vote(int(people.tracker_id[i]), team)
            teams[i] = self._team_of(int(people.tracker_id[i]))
