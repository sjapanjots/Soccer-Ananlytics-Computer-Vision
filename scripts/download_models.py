"""Download the pretrained football models (and optionally sample videos).

The weights come from Roboflow's open-source ``roboflow/sports`` project
(https://github.com/roboflow/sports, MIT licence) and are hosted on Google
Drive. No training is needed: download once and reuse for every video.

Usage:
    python scripts/download_models.py            # models only
    python scripts/download_models.py --videos   # models + 5 sample clips
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
VIDEOS_DIR = ROOT / "videos"

MODELS = {
    "football-player-detection.pt": "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q",
    "football-ball-detection.pt": "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V",
    "football-pitch-detection.pt": "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf",
}

SAMPLE_VIDEOS = {
    "0bfacc_0.mp4": "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF",
    "2e57b9_0.mp4": "19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf",
    "08fd33_0.mp4": "1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-",
    "573e61_0.mp4": "1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU",
    "121364_0.mp4": "1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu",
}


def download(files: dict[str, str], target: Path, force: bool) -> bool:
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is not installed. Run: pip install gdown")

    target.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, file_id in files.items():
        path = target / name
        if path.exists() and path.stat().st_size > 0 and not force:
            print(f"[skip] {path.relative_to(ROOT)} already exists")
            continue
        print(f"[download] {name}")
        result = gdown.download(id=file_id, output=str(path), quiet=False)
        if result is None or not path.exists():
            print(f"[error] failed to download {name}", file=sys.stderr)
            ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", action="store_true", help="also download sample match clips")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    ok = download(MODELS, MODELS_DIR, args.force)
    if args.videos:
        ok = download(SAMPLE_VIDEOS, VIDEOS_DIR, args.force) and ok
    if not ok:
        sys.exit(
            "Some downloads failed. Google Drive sometimes rate-limits; "
            "wait a while and run the script again."
        )
    print("Done.")


if __name__ == "__main__":
    main()
