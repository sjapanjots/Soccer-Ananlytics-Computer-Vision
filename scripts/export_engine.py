"""Export a YOLO model to TensorRT (FP16) for the fastest NVIDIA inference.

On a 4 GB laptop GPU such as the RTX 2050 this typically gives a 1.5-2x speed
up over plain PyTorch FP16. The engine is tied to the GPU, TensorRT version,
image size and batch size it was built with, so build it on the machine that
will run the app and use the same values in the app settings.

Usage:
    pip install tensorrt   # Linux; on Windows install TensorRT from NVIDIA
    python scripts/export_engine.py --model models/football-player-detection.pt

Then point the app's "Player model" setting (or ``--player-model`` on the CLI)
at ``models/football-player-detection.engine``.
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO weights to TensorRT")
    parser.add_argument("--model", default="models/football-player-detection.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2, help="must match the app's batch size")
    parser.add_argument("--workspace", type=float, default=2.0, help="TensorRT workspace in GB")
    args = parser.parse_args()

    model = YOLO(args.model)
    path = model.export(
        format="engine",
        half=True,
        imgsz=args.imgsz,
        batch=args.batch,
        workspace=args.workspace,
        device=0,
    )
    print(f"Saved TensorRT engine to {path}")


if __name__ == "__main__":
    main()
