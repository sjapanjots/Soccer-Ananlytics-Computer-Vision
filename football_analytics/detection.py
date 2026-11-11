"""Object detection with Ultralytics YOLO.

Works with:
* Roboflow's pretrained ``football-player-detection.pt`` (ball, goalkeeper,
  player, referee) - recommended, no training required;
* an optional dedicated ``football-ball-detection.pt`` for better ball recall;
* any COCO model (e.g. ``yolo11n.pt``) as a fallback, mapping "person" to
  player and "sports ball" to ball, so the app runs before the football
  weights are downloaded;
* TensorRT ``.engine`` exports of the above (see ``scripts/export_engine.py``).

Every model's own class ids are normalised to the ids in ``config``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import supervision as sv
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

from .config import (
    BALL_CLASS_ID,
    FALLBACK_MODEL,
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    REFEREE_CLASS_ID,
    Settings,
    resolve_device,
)

log = logging.getLogger(__name__)

# Ultralytics >= 8.4 replaced ``half=True`` with ``quantize=16``.
_HAS_QUANTIZE = "quantize" in DEFAULT_CFG_DICT


def _precision_kwargs(half: bool) -> dict:
    if _HAS_QUANTIZE:
        return {"quantize": 16} if half else {}
    return {"half": half}

# Model class name -> normalised class id.
NAME_TO_CLASS = {
    "ball": BALL_CLASS_ID,
    "sports ball": BALL_CLASS_ID,
    "goalkeeper": GOALKEEPER_CLASS_ID,
    "player": PLAYER_CLASS_ID,
    "person": PLAYER_CLASS_ID,
    "referee": REFEREE_CLASS_ID,
}


@dataclass
class FrameDetections:
    """Detections for one frame, split into people and ball candidates."""

    people: sv.Detections  # players, goalkeepers and referees
    ball: sv.Detections  # zero or more ball candidates


def _load_model(path: str) -> YOLO:
    task = "detect" if Path(path).suffix in {".engine", ".onnx", ".xml"} else None
    return YOLO(path, task=task)


def _class_map(model: YOLO) -> dict[int, int]:
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    return {
        int(idx): NAME_TO_CLASS[name.lower()]
        for idx, name in names.items()
        if name.lower() in NAME_TO_CLASS
    }


def _normalise(detections: sv.Detections, class_map: dict[int, int]) -> sv.Detections:
    if len(detections) == 0:
        return detections
    keep = np.isin(detections.class_id, list(class_map))
    detections = detections[keep]
    detections.class_id = np.array(
        [class_map[int(c)] for c in detections.class_id], dtype=int
    )
    return detections


class Detector:
    """Runs the player model (and optionally the ball model) on frame batches."""

    def __init__(
        self,
        player_model: str,
        ball_model: str | None = None,
        device: str = "auto",
        half: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        self.half = half and self.device.startswith("cuda")

        self.using_fallback = not Path(player_model).exists()
        if self.using_fallback:
            log.warning(
                "Football model %s not found; falling back to %s (generic COCO "
                "person / sports ball). Run scripts/download_models.py for much "
                "better results.",
                player_model,
                FALLBACK_MODEL,
            )
            player_model = FALLBACK_MODEL
        self.player_model_path = player_model
        self.player_model = _load_model(player_model)
        self.player_classes = _class_map(self.player_model)
        if not self.player_classes:
            raise ValueError(
                f"{player_model} has none of the expected classes "
                f"{sorted(NAME_TO_CLASS)}; got {self.player_model.names}"
            )

        self.ball_model = None
        self.ball_classes: dict[int, int] = {}
        if ball_model and Path(ball_model).exists():
            self.ball_model = _load_model(ball_model)
            self.ball_classes = {
                k: v for k, v in _class_map(self.ball_model).items() if v == BALL_CLASS_ID
            }

    @classmethod
    def from_settings(cls, settings: Settings) -> "Detector":
        return cls(
            player_model=settings.player_model,
            ball_model=settings.ball_model if settings.use_ball_model else None,
            device=settings.device,
            half=settings.half,
        )

    @property
    def has_ball_model(self) -> bool:
        return self.ball_model is not None

    def _predict(self, model: YOLO, frames: Sequence[np.ndarray], conf: float, imgsz: int):
        return model.predict(
            list(frames),
            imgsz=imgsz,
            conf=conf,
            device=self.device,
            **_precision_kwargs(self.half),
            verbose=False,
        )

    def detect(
        self,
        frames: Sequence[np.ndarray],
        settings: Settings,
        use_ball_model: bool | None = None,
    ) -> list[FrameDetections]:
        """Detect people and ball candidates in a batch of BGR frames."""
        if not frames:
            return []
        if use_ball_model is None:
            use_ball_model = settings.use_ball_model
        use_ball_model = use_ball_model and self.has_ball_model

        min_conf = min(settings.player_conf, settings.ball_conf)
        results = self._predict(self.player_model, frames, min_conf, settings.imgsz)

        ball_results = None
        if use_ball_model:
            ball_results = self._predict(
                self.ball_model, frames, settings.ball_conf, settings.imgsz
            )

        output = []
        for i, result in enumerate(results):
            detections = _normalise(sv.Detections.from_ultralytics(result), self.player_classes)
            is_ball = detections.class_id == BALL_CLASS_ID

            people = detections[~is_ball]
            people = people[people.confidence >= settings.player_conf]
            people = people.with_nms(threshold=settings.nms_iou, class_agnostic=True)

            if ball_results is not None:
                ball = _normalise(
                    sv.Detections.from_ultralytics(ball_results[i]), self.ball_classes
                )
            else:
                ball = detections[is_ball]
            ball = ball[ball.confidence >= settings.ball_conf]
            output.append(FrameDetections(people=people, ball=ball))
        return output
