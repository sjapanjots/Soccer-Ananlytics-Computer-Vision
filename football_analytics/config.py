"""Shared settings and constants for the analytics pipeline.

The defaults are tuned for a laptop GPU with ~4 GB of VRAM (e.g. an NVIDIA
RTX 2050): FP16 inference, small batches and a frame stride of 2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
VIDEOS_DIR = ROOT_DIR / "videos"
OUTPUTS_DIR = ROOT_DIR / "outputs"

PLAYER_MODEL_PATH = MODELS_DIR / "football-player-detection.pt"
BALL_MODEL_PATH = MODELS_DIR / "football-ball-detection.pt"
PITCH_MODEL_PATH = MODELS_DIR / "football-pitch-detection.pt"

# Used when the Roboflow football weights have not been downloaded yet.
# Ultralytics downloads it automatically on first use (COCO classes).
FALLBACK_MODEL = "yolo11n.pt"

# Normalised class ids used everywhere in the pipeline. They match the class
# order of Roboflow's football-player-detection model.
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3
CLASS_NAMES = {
    BALL_CLASS_ID: "ball",
    GOALKEEPER_CLASS_ID: "goalkeeper",
    PLAYER_CLASS_ID: "player",
    REFEREE_CLASS_ID: "referee",
}

# Team ids. Players and goalkeepers get 0 or 1, everyone else NO_TEAM.
TEAM_A = 0
TEAM_B = 1
NO_TEAM = -1


@dataclass(frozen=True)
class Settings:
    """All tunable knobs of the pipeline. Immutable; use ``with_overrides``."""

    # --- models / hardware -------------------------------------------------
    player_model: str = str(PLAYER_MODEL_PATH)
    ball_model: str = str(BALL_MODEL_PATH)
    use_ball_model: bool = False  # extra, more accurate ball detector (slower)
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", ...
    half: bool = True  # FP16 on CUDA; ignored on CPU

    # --- speed -------------------------------------------------------------
    imgsz: int = 1280  # Roboflow football models were trained at 1280
    batch_size: int = 2  # frames per forward pass; 2 fits in 4 GB at 1280
    frame_stride: int = 2  # run detection on every Nth frame

    # --- detection ---------------------------------------------------------
    player_conf: float = 0.3
    ball_conf: float = 0.25
    nms_iou: float = 0.5

    # --- teams -------------------------------------------------------------
    team_calibration_frames: int = 20  # frames sampled across the video
    team_vote_window: int = 15  # rolling majority vote per tracked player

    # --- events ------------------------------------------------------------
    # The ball belongs to the nearest player if it is closer than
    # ``possession_distance_ratio`` * that player's box height. Relative to
    # the player's size so it works at any camera zoom or video resolution.
    possession_distance_ratio: float = 0.5
    possession_confirm_frames: int = 3  # processed frames to confirm owner

    # --- rendering ---------------------------------------------------------
    draw_possession: bool = True
    draw_passes: bool = True
    ball_trail_length: int = 20
    pass_arrow_seconds: float = 1.5

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)


# Speed / quality presets shown in the UI and CLI.
PRESETS: dict[str, dict] = {
    "Quality": {"imgsz": 1280, "frame_stride": 1, "batch_size": 2},
    "Balanced": {"imgsz": 1280, "frame_stride": 2, "batch_size": 2},
    "Fast": {"imgsz": 960, "frame_stride": 3, "batch_size": 4},
}
DEFAULT_PRESET = "Balanced"


def resolve_device(device: str = "auto") -> str:
    """Return a concrete torch device string ("cuda:0" or "cpu")."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass
    return "cpu"


def device_summary() -> str:
    """Human readable description of the compute device, for the UI/logs."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1024**3
            return f"{props.name} ({vram_gb:.1f} GB VRAM, CUDA {torch.version.cuda})"
    except ImportError:  # pragma: no cover
        pass
    return "CPU only (no CUDA GPU detected)"
