"""Draws the analysis on top of the video frames (OpenCV + supervision).

* Ellipse under each person in their team's actual kit colour
  (referees yellow, unassigned grey) with the tracker id.
* Red triangle above the ball, plus a fading ball trail.
* Small marker above the player currently in possession.
* Arrow for each recent pass (team colour) or turnover (red).
* Scoreboard (bottom-left) with the possession bar and pass counts.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import supervision as sv

from .config import NO_TEAM, REFEREE_CLASS_ID, Settings
from .events import BallEvent, MatchStats

REFEREE_BGR = (0, 215, 255)  # yellow
UNKNOWN_BGR = (160, 160, 160)  # grey
BALL_BGR = (0, 0, 255)  # red
OWNER_BGR = (0, 0, 255)
TURNOVER_BGR = (60, 20, 220)
PANEL_BGR = (25, 25, 25)

_REFEREE_INDEX = 2
_UNKNOWN_INDEX = 3


def _text_color_for(bgr: tuple[int, int, int]) -> sv.Color:
    b, g, r = bgr
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return sv.Color.BLACK if luminance > 150 else sv.Color.WHITE


def _visible(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Brighten / saturate a kit colour so it stands out on grass.

    Median shirt colours are muted by shadows and compression (a navy kit
    would be almost invisible); keep the hue but raise brightness and
    saturation. Near-white and near-black kits keep their low saturation.
    """
    pixel = np.uint8([[bgr]])
    h, s, v = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0].astype(int)
    if s > 40:
        s = min(255, max(s * 3 // 2, 140))
    v = max(v, 210)
    b, g, r = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


class FrameAnnotator:
    def __init__(
        self,
        team_colors_bgr: list[tuple[int, int, int]],
        frame_size: tuple[int, int],
        fps: float,
        settings: Settings,
        team_names: tuple[str, str] = ("Team A", "Team B"),
    ) -> None:
        self.settings = settings
        self.team_names = team_names
        width, _ = frame_size
        self.scale = max(width / 1920, 0.4)
        self.fps = fps if fps > 0 else 25.0
        self.set_team_colors(team_colors_bgr)

        size = max(8, round(18 * self.scale))
        self._ball = sv.TriangleAnnotator(
            color=sv.Color.from_bgr_tuple(BALL_BGR),
            base=size,
            height=size,
            outline_thickness=1,
        )
        self._owner = sv.TriangleAnnotator(
            color=sv.Color.from_bgr_tuple(OWNER_BGR),
            base=round(size * 0.8),
            height=round(size * 0.8),
            outline_thickness=1,
        )
        self._trail: deque[tuple[int, int] | None] = deque(maxlen=settings.ball_trail_length)
        self._arrows: deque[tuple[int, BallEvent]] = deque(maxlen=8)

    # ------------------------------------------------------------------ state
    def set_team_colors(self, team_colors_bgr: list[tuple[int, int, int]]) -> None:
        """(Re)build the per-team palettes, e.g. once teams are calibrated."""
        self.team_colors = [_visible(c) for c in team_colors_bgr]
        palette_bgr = [*self.team_colors, REFEREE_BGR, UNKNOWN_BGR]
        palette = sv.ColorPalette([sv.Color.from_bgr_tuple(c) for c in palette_bgr])
        text_palette = sv.ColorPalette([_text_color_for(c) for c in palette_bgr])
        self._ellipse = sv.EllipseAnnotator(color=palette, thickness=max(1, round(2 * self.scale)))
        self._label = sv.LabelAnnotator(
            color=palette,
            text_color=text_palette,
            text_position=sv.Position.BOTTOM_CENTER,
            text_scale=0.45 * self.scale,
            text_thickness=1,
            text_padding=max(2, round(4 * self.scale)),
        )

    def update_ball_trail(self, ball: sv.Detections) -> None:
        if len(ball):
            x, y = ball.get_anchors_coordinates(sv.Position.CENTER)[0]
            self._trail.append((int(x), int(y)))
        else:
            self._trail.append(None)

    def add_event(self, frame_index: int, event: BallEvent) -> None:
        self._arrows.append((frame_index, event))

    # ------------------------------------------------------------------- draw
    def draw(
        self,
        frame: np.ndarray,
        frame_index: int,
        people: sv.Detections,
        teams: np.ndarray,
        ball: sv.Detections,
        stats: MatchStats | None,
    ) -> np.ndarray:
        scene = frame.copy()
        if len(people):
            lookup = np.where(teams == NO_TEAM, _UNKNOWN_INDEX, teams)
            is_referee = (people.class_id == REFEREE_CLASS_ID) & (teams == NO_TEAM)
            lookup = np.where(is_referee, _REFEREE_INDEX, lookup).astype(int)
            labels = [f"#{tid}" for tid in people.tracker_id]
            scene = self._ellipse.annotate(scene, people, custom_color_lookup=lookup)
            scene = self._label.annotate(scene, people, labels, custom_color_lookup=lookup)

            if stats is not None and stats.owner is not None:
                owner = people[people.tracker_id == stats.owner.tracker_id]
                if len(owner):
                    scene = self._owner.annotate(scene, owner)

        self._draw_trail(scene)
        if len(ball):
            scene = self._ball.annotate(scene, ball)

        if stats is not None:
            if self.settings.draw_passes:
                self._draw_arrows(scene, frame_index)
            self._draw_scoreboard(scene, stats)
        return scene

    def _draw_trail(self, scene: np.ndarray) -> None:
        points = list(self._trail)
        n = len(points)
        for i in range(1, n):
            if points[i - 1] is None or points[i] is None:
                continue
            thickness = max(1, round((1 + 3 * i / n) * self.scale))
            cv2.line(scene, points[i - 1], points[i], BALL_BGR, thickness, cv2.LINE_AA)

    def _draw_arrows(self, scene: np.ndarray, frame_index: int) -> None:
        max_age = self.settings.pass_arrow_seconds * self.fps
        thickness = max(2, round(3 * self.scale))
        for created, event in self._arrows:
            if frame_index - created > max_age:
                continue
            color = self.team_colors[event.team] if event.kind == "pass" else TURNOVER_BGR
            start = tuple(int(v) for v in event.start_xy)
            end = tuple(int(v) for v in event.end_xy)
            cv2.arrowedLine(scene, start, end, (0, 0, 0), thickness + 2, cv2.LINE_AA, tipLength=0.08)
            cv2.arrowedLine(scene, start, end, color, thickness, cv2.LINE_AA, tipLength=0.08)

    def _draw_scoreboard(self, scene: np.ndarray, stats: MatchStats) -> None:
        s = self.scale
        w, row = round(500 * s), round(44 * s)
        rows = 1 + int(self.settings.draw_possession) + int(self.settings.draw_passes)
        h = row * rows + round(16 * s)
        # Bottom-left: broadcasters put their own score graphic at the top.
        x0, y0 = round(20 * s), scene.shape[0] - h - round(20 * s)

        overlay = scene.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), PANEL_BGR, -1)
        cv2.addWeighted(overlay, 0.7, scene, 0.3, 0, dst=scene)

        font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.75 * s, max(1, round(2 * s))
        pad = round(12 * s)
        sw = round(22 * s)
        y = y0 + pad

        # Team names with colour swatches.
        half = w // 2
        for team, name in enumerate(self.team_names):
            x = x0 + pad + team * half
            cv2.rectangle(scene, (x, y), (x + sw, y + sw), self.team_colors[team], -1)
            cv2.rectangle(scene, (x, y), (x + sw, y + sw), (255, 255, 255), 1)
            cv2.putText(scene, name, (x + sw + pad // 2, y + sw - 2), font, fs, (255, 255, 255), ft, cv2.LINE_AA)
        y += row

        if self.settings.draw_possession:
            pct = stats.possession_pct
            bar_w = w - 2 * pad
            split = round(bar_w * pct[0] / 100) if sum(pct) > 0 else bar_w // 2
            bx, bh = x0 + pad, round(28 * s)
            cv2.rectangle(scene, (bx, y), (bx + split, y + bh), self.team_colors[0], -1)
            cv2.rectangle(scene, (bx + split, y), (bx + bar_w, y + bh), self.team_colors[1], -1)
            cv2.rectangle(scene, (bx, y), (bx + bar_w, y + bh), (255, 255, 255), 1)
            for team, align_left in ((0, True), (1, False)):
                text = f"{pct[team]:.0f}%"
                (tw, th), _ = cv2.getTextSize(text, font, fs, ft)
                tx = bx + pad // 2 if align_left else bx + bar_w - tw - pad // 2
                color = _text_color_for(self.team_colors[team]).as_bgr()
                cv2.putText(scene, text, (tx, y + (bh + th) // 2), font, fs, color, ft, cv2.LINE_AA)
            y += row

        if self.settings.draw_passes:
            passes = stats.passes
            for team in range(2):
                x = x0 + pad + team * half
                cv2.putText(scene, f"Passes: {passes[team]}", (x, y + sw - 2), font, fs, (255, 255, 255), ft, cv2.LINE_AA)
