"""Video reading (OpenCV) and browser-friendly writing (FFmpeg, H.264).

OpenCV's own ``mp4v`` output does not play in browsers, so Streamlit's video
player would stay blank. Frames are instead piped straight into FFmpeg and
encoded once as H.264 / yuv420p with ``+faststart``. FFmpeg is taken from the
PATH or, if missing (common on Windows), from the ``imageio-ffmpeg`` wheel.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    total_frames: int

    @property
    def duration_s(self) -> float:
        return self.total_frames / self.fps if self.fps else 0.0


def get_video_info(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        return VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(fps),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def iter_frames(path: str | Path, max_frames: int | None = None) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    try:
        count = 0
        while max_frames is None or count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            count += 1
    finally:
        cap.release()


def sample_frames(path: str | Path, count: int, max_frames: int | None = None) -> list[np.ndarray]:
    """``count`` frames spread evenly over the video (or its first ``max_frames``)."""
    info = get_video_info(path)
    total = info.total_frames if max_frames is None else min(info.total_frames, max_frames)
    if total <= 0 or count <= 0:
        return []
    indices = np.linspace(0, total - 1, num=min(count, total), dtype=int)
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    finally:
        cap.release()
    return frames


def ffmpeg_executable() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FFmpeg not found. Install it (https://ffmpeg.org) or `pip install imageio-ffmpeg`."
        ) from exc


class H264Writer:
    """Writes BGR frames to an H.264 MP4 through an FFmpeg pipe.

    ``encoder`` can be ``libx264`` (CPU, always available) or ``h264_nvenc``
    (NVIDIA hardware encoder, if your FFmpeg build has it).
    """

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        encoder: str = "libx264",
        crf: int = 23,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (width, height)
        quality = ["-preset", "veryfast", "-crf", str(crf)]
        if encoder == "h264_nvenc":
            quality = ["-preset", "p4", "-cq", str(crf)]
        cmd = [
            ffmpeg_executable(),
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", f"{fps:.6f}",
            "-i", "-",
            "-an",
            # yuv420p needs even dimensions.
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", encoder,
            *quality,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(self.path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(f"FFmpeg failed: {self._stderr()}") from exc

    def _stderr(self) -> str:
        return self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""

    def close(self) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        code = self._proc.wait()
        if code != 0:
            raise RuntimeError(f"FFmpeg exited with code {code}: {self._stderr()}")

    def __enter__(self) -> "H264Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self._proc.kill()
            self._proc.wait()
