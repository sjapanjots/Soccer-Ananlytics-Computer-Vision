"""End-to-end video analysis: detect -> track -> teams -> events -> draw.

Speed tricks (all configurable through ``Settings``):
* one model for players, goalkeepers, referees and ball;
* FP16 inference on CUDA, batched frames;
* detection only on every ``frame_stride``-th frame, the in-between frames
  reuse the latest overlay (possession time is weighted accordingly);
* team colour is fitted once and cached per tracked player;
* single-pass H.264 encoding through an FFmpeg pipe.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import supervision as sv

from .annotate import FrameAnnotator
from .config import PLAYER_CLASS_ID, Settings
from .detection import Detector
from .events import BallEvent, MatchStats
from .team import TeamAssigner, extract_features
from .tracking import BallTracker, PeopleTracker
from .video import H264Writer, get_video_info, iter_frames, sample_frames

# Shown until the team colours are known.
PLACEHOLDER_TEAM_COLORS = [(230, 230, 230), (200, 120, 30)]

# progress(frames_done, frames_total, processing_fps)
ProgressCallback = Callable[[int, int, float], None]


@dataclass
class AnalysisResult:
    output_video: Path
    summary: dict
    events: list[BallEvent] = field(default_factory=list)
    timeline: list[tuple[int, float, int]] = field(default_factory=list)

    def events_frame(self) -> pd.DataFrame:
        columns = ["kind", "frame", "time_s", "team", "from_player", "to_player", "start_xy", "end_xy"]
        return pd.DataFrame([e.to_dict() for e in self.events], columns=columns)

    def timeline_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.timeline, columns=["frame", "time_s", "team"])


def _hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def calibrate_teams(
    detector: Detector,
    video_path: str | Path,
    settings: Settings,
    assigner: TeamAssigner,
    max_frames: int | None = None,
) -> bool:
    """Fit the team colours on players sampled across the whole video."""
    frames = sample_frames(video_path, settings.team_calibration_frames, max_frames)
    features = []
    for start in range(0, len(frames), settings.batch_size):
        batch = frames[start : start + settings.batch_size]
        for frame, detections in zip(batch, detector.detect(batch, settings, use_ball_model=False)):
            players = detections.people[detections.people.class_id == PLAYER_CLASS_ID]
            feats, _ = extract_features(frame, players)
            features.extend(feats)
    if not features:
        return False
    return assigner.calibrate(np.stack(features))


def analyze_video(
    input_path: str | Path,
    output_path: str | Path,
    settings: Settings | None = None,
    detector: Detector | None = None,
    progress: ProgressCallback | None = None,
    max_frames: int | None = None,
    encoder: str = "libx264",
) -> AnalysisResult:
    settings = settings or Settings()
    detector = detector or Detector.from_settings(settings)
    stride = max(1, settings.frame_stride)
    batch_size = max(1, settings.batch_size)

    info = get_video_info(input_path)
    total = info.total_frames if max_frames is None else min(info.total_frames, max_frames)
    started = time.perf_counter()

    assigner = TeamAssigner(vote_window=settings.team_vote_window)
    calibrate_teams(detector, input_path, settings, assigner, max_frames)
    colors = assigner.classifier.team_colors_bgr if assigner.ready else PLACEHOLDER_TEAM_COLORS

    people_tracker = PeopleTracker(frame_rate=info.fps / stride)
    ball_tracker = BallTracker((info.width, info.height))
    stats = MatchStats(info.fps, settings)
    annotator = FrameAnnotator(colors, (info.width, info.height), info.fps, settings)

    empty = sv.Detections.empty()
    last_state = (empty, np.empty(0, dtype=int), empty)
    frame_index = 0
    chunk: list[np.ndarray] = []
    chunk_len = batch_size * stride

    def flush(writer: H264Writer) -> None:
        nonlocal frame_index, last_state
        to_detect = [f for i, f in enumerate(chunk) if (frame_index + i) % stride == 0]
        detections = iter(detector.detect(to_detect, settings))
        for frame in chunk:
            if frame_index % stride == 0:
                found = next(detections)
                was_ready = assigner.ready
                people = people_tracker.update(found.people)
                teams = assigner.assign(frame, people)
                if not was_ready and assigner.ready:
                    annotator.set_team_colors(assigner.classifier.team_colors_bgr)
                ball = ball_tracker.update(found.ball)
                event = stats.update(frame_index, people, teams, ball, weight=stride)
                if event is not None:
                    annotator.add_event(frame_index, event)
                annotator.update_ball_trail(ball)
                last_state = (people, teams, ball)
            writer.write(annotator.draw(frame, frame_index, *last_state, stats))
            frame_index += 1
        chunk.clear()
        if progress:
            elapsed = time.perf_counter() - started
            progress(frame_index, total, frame_index / elapsed if elapsed else 0.0)

    with H264Writer(output_path, info.width, info.height, info.fps, encoder=encoder) as writer:
        for frame in iter_frames(input_path, max_frames):
            chunk.append(frame)
            if len(chunk) == chunk_len:
                flush(writer)
        if chunk:
            flush(writer)

    elapsed = time.perf_counter() - started
    team_colors = assigner.classifier.team_colors_bgr if assigner.ready else PLACEHOLDER_TEAM_COLORS
    summary = {
        **stats.summary(),
        "team_colors": [_hex(c) for c in team_colors],
        "teams_detected": assigner.ready,
        "frames": frame_index,
        "video_fps": round(info.fps, 3),
        "video_duration_s": round(frame_index / info.fps, 2) if info.fps else 0.0,
        "resolution": [info.width, info.height],
        "processing_s": round(elapsed, 2),
        "processing_fps": round(frame_index / elapsed, 2) if elapsed else 0.0,
        "device": detector.device,
        "model": detector.player_model_path,
        "fallback_model": detector.using_fallback,
        "settings": settings.to_dict(),
    }
    return AnalysisResult(
        output_video=Path(output_path),
        summary=summary,
        events=list(stats.events),
        timeline=list(stats.timeline),
    )


def export_results(result: AnalysisResult, out_dir: str | Path) -> dict[str, Path]:
    """Write summary.json, events.csv and possession_timeline.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": out_dir / "summary.json",
        "events": out_dir / "events.csv",
        "timeline": out_dir / "possession_timeline.csv",
    }
    paths["summary"].write_text(json.dumps(result.summary, indent=2))
    result.events_frame().to_csv(paths["events"], index=False)
    result.timeline_frame().to_csv(paths["timeline"], index=False)
    return paths


__all__ = [
    "AnalysisResult",
    "ProgressCallback",
    "analyze_video",
    "calibrate_teams",
    "export_results",
]
