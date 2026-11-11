"""Command-line entry point.

Example:
    python analyze.py --video videos/0bfacc_0.mp4
    python analyze.py --video match.mp4 --preset Fast --ball-model

Writes ``outputs/<video name>/annotated.mp4`` plus summary.json, events.csv
and possession_timeline.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from football_analytics.config import DEFAULT_PRESET, OUTPUTS_DIR, PRESETS, Settings, device_summary
from football_analytics.pipeline import analyze_video, export_results


def parse_args() -> argparse.Namespace:
    defaults = Settings()
    p = argparse.ArgumentParser(description="Soccer video analytics")
    p.add_argument("--video", required=True, help="input video file")
    p.add_argument("--output-dir", help="default: outputs/<video name>")
    p.add_argument("--preset", choices=list(PRESETS), default=DEFAULT_PRESET)
    p.add_argument("--player-model", default=defaults.player_model)
    p.add_argument("--ball-model", action="store_true", help="also run the dedicated ball model")
    p.add_argument("--device", default="auto", help='"auto", "cpu", "cuda:0"...')
    p.add_argument("--no-half", action="store_true", help="disable FP16")
    p.add_argument("--imgsz", type=int, help="override the preset's inference size")
    p.add_argument("--stride", type=int, help="override the preset's frame stride")
    p.add_argument("--batch", type=int, help="override the preset's batch size")
    p.add_argument("--conf", type=float, default=defaults.player_conf, help="player confidence")
    p.add_argument("--max-frames", type=int, help="only process the first N frames")
    p.add_argument("--encoder", default="libx264", choices=["libx264", "h264_nvenc"])
    p.add_argument("--no-possession", action="store_true")
    p.add_argument("--no-passes", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    overrides = dict(PRESETS[args.preset])
    for key, value in (("imgsz", args.imgsz), ("frame_stride", args.stride), ("batch_size", args.batch)):
        if value is not None:
            overrides[key] = value
    settings = Settings().with_overrides(
        **overrides,
        player_model=args.player_model,
        use_ball_model=args.ball_model,
        device=args.device,
        half=not args.no_half,
        player_conf=args.conf,
        draw_possession=not args.no_possession,
        draw_passes=not args.no_passes,
    )

    video = Path(args.video)
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / video.stem
    print(f"Device: {device_summary()}")

    bar = tqdm(unit="frame")

    def progress(done: int, total: int, fps: float) -> None:
        bar.total = total
        bar.n = done
        bar.set_postfix(fps=f"{fps:.1f}")
        bar.refresh()

    result = analyze_video(
        video,
        out_dir / "annotated.mp4",
        settings,
        progress=progress,
        max_frames=args.max_frames,
        encoder=args.encoder,
    )
    bar.close()
    export_results(result, out_dir)

    summary = {k: v for k, v in result.summary.items() if k != "settings"}
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
