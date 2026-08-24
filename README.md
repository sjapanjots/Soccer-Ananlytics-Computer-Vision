# Soccer Analytics — Computer Vision ⚽

AI-powered soccer match analysis straight from raw video footage. This project uses **YOLOv5** object detection, **Norfair** multi-object tracking, and **HSV color classification** to automatically compute **ball possession**, detect **pass events**, and visualize everything live on top of the video.

<p align="center">
  <img src="images/thumbnail.png" alt="Soccer Analytics Demo" width="720"/>
</p>

---

## 📋 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Customization](#-customization)
- [Tech Stack](#-tech-stack)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Player & Ball Detection** | Detects players with a COCO-pretrained `yolov5x` model and the ball with a custom-trained YOLOv5 model |
| 🔭 **Multi-Object Tracking** | Keeps consistent player/ball identities across frames using Norfair trackers |
| 🎥 **Camera Motion Compensation** | Estimates camera motion every frame so tracking stays stable even when the camera pans |
| 👕 **Team Classification** | Assigns each player to a team (or referee) by classifying jersey colors in HSV space |
| 🧊 **Classification Inertia** | Smooths out noisy per-frame classifications with an inertia-based wrapper |
| 📊 **Possession Counter** | Live on-screen scoreboard showing which team currently has possession |
| 🔄 **Pass Detection** | Detects pass events between teammates and draws them on the field |
| 🛤️ **Ball Trajectory** | Draws the ball's path, colored by the team in possession |
| 🆔 **Player IDs** | Overlays unique tracker IDs on each detected player |

---

## 🧠 How It Works

The pipeline processes the input video frame by frame:

```
Video Frame
    │
    ├──► YOLOv5 (COCO weights) ──► Player detections   (confidence > 0.35)
    │
    ├──► YOLOv5 (custom ball.pt) ─► Ball detections     (confidence > 0.3)
    │
    ▼
Camera Motion Estimation (Norfair MotionEstimator)
    │
    ▼
Tracking (Norfair Tracker — mean euclidean distance)
    │
    ▼
HSV Jersey Classification ──► Inertia Smoothing ──► Team assignment
    │
    ▼
Match Logic
    ├──► Closest player to ball → possession change detection
    ├──► Possession counter (per team)
    └──► Pass event detection
    │
    ▼
Overlay Rendering (PIL) ──► Output annotated video
```

### Key components

- **Detection** (`inference/yolov5.py`): Loads models through `torch.hub`. Without a custom model path it downloads `yolov5x` with COCO weights (used to find `person` detections); with `--model models/ball.pt` it loads a custom-trained ball detector.
- **Team classification** (`inference/hsv_classifier.py`, `inertia_classifier.py`): Each player crop is converted to HSV and matched against configured jersey color filters; an inertia layer prevents rapid identity flickering between frames.
- **Match logic** (`soccer/match.py`): Tracks the player closest to the ball. A team must hold possession for a configurable number of consecutive frames before the possession state changes, avoiding false switches.
- **Pass events** (`soccer/pass_event.py`): Detects when the ball moves from one player to another teammate and records it as a pass.
- **Drawing** (`soccer/draw.py`): Renders trajectories, pass lines, possession/pass boards, IDs, and colors onto frames using absolute (camera-compensated) coordinates.

---

## 📁 Project Structure

```
Soccer-Ananlytics-Computer-Vision/
├── inference/            # Detection & classification modules
│   ├── yolov5.py         # YOLOv5 detector wrapper (torch.hub)
│   ├── hsv_classifier.py # HSV-based jersey color classifier
│   ├── inertia_classifier.py # Adds temporal inertia to classifications
│   ├── nn_classifier.py  # Neural network classifier (optional)
│   ├── filters.py        # Team/referee color filter definitions
│   ├── base_detector.py / base_classifier.py
│   ├── converter.py      # DataFrame ↔ Norfair Detection conversions
│   ├── box.py / colors.py
├── soccer/               # Domain logic
│   ├── match.py          # Match state: possession, closest player, counters
│   ├── player.py         # Player entity (feet points, distance to ball, drawing)
│   ├── team.py           # Team entity (name, abbreviation, colors)
│   ├── ball.py           # Ball entity
│   ├── pass_event.py     # Pass / PassEvent logic and drawing
│   └── draw.py           # Trajectory & overlay drawing utilities
├── fonts/                # Font used for overlay text (Gidole-Regular.ttf)
├── images/               # Background boards for possession/pass counters
├── run.py                # Main entry point
├── run_utils.py          # Detection filtering & motion estimation helpers
└── pyproject.toml        # Poetry project configuration
```

---

## ✅ Requirements

- **Python 3.9+**
- **Poetry** (dependency manager)
- **NVIDIA GPU with CUDA** strongly recommended (inference runs on CPU otherwise, and will be slow)
- Input video file (e.g., `videos/soccer_possession.mp4`)
- Custom trained ball detection model (e.g., `models/ball.pt`)

> ⚠️ Note: `*.mp4` and `*.pt` files are gitignored — you must supply your own video and ball model, or download them separately.

---

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/Soccer-Ananlytics-Computer-Vision.git
   cd Soccer-Ananlytics-Computer-Vision
   ```

2. **Install Poetry** (if not already installed)

   ```bash
   pip install poetry
   ```

3. **Install dependencies**

   ```bash
   poetry install
   ```

4. **Activate the virtual environment**

   ```bash
   poetry shell
   ```

---

## 🚀 Usage

Run the full pipeline (possession + passes):

```bash
python run.py --video videos/soccer_possession.mp4 --model models/ball.pt --possession --passes
```

Enable only specific features:

```bash
# Possession counter only
python run.py --video videos/soccer_possession.mp4 --model models/ball.pt --possession

# Pass detection only
python run.py --video videos/soccer_possession.mp4 --model models/ball.pt --passes
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | `videos/soccer_possession.mp4` | Path to the input video |
| `--model` | `models/ball.pt` | Path to the custom YOLOv5 ball detection model |
| `--possession` | off | Enable the possession counter overlay |
| `--passes` | off | Enable the pass detection overlay |

The annotated output video is written next to the input file.

---

## ⚙️ Customization

### Add or change teams

Edit the team definitions in [`run.py`](run.py):

```python
chelsea = Team(
    name="Chelsea",
    abbreviation="CHE",
    color=(255, 0, 0),
    board_color=(244, 86, 64),
    text_color=(255, 255, 255),
)
man_city = Team(name="Man City", abbreviation="MNC", color=(240, 230, 188))
teams = [chelsea, man_city]
```

Then update the corresponding jersey color filters in [`inference/filters.py`](inference/filters.py) so the HSV classifier recognizes the new kits:

```python
filters = [
    chelsea_filter,   # {"name": ..., "colors": [...]}
    city_filter,
    referee_filter,
]
```

### Tune match behavior

In [`soccer/match.py`](soccer/match.py) you can adjust:

- `possesion_counter_threshold` — consecutive frames required before possession changes
- `ball_distance_threshold` — max pixel distance between a player's foot and the ball to count as control

Tracker sensitivity (distance thresholds, hit counters, initialization delay) is configured in [`run.py`](run.py).

---

## 💻 Tech Stack

- [PyTorch](https://pytorch.org/) + [torchvision](https://pytorch.org/vision/) — deep learning inference
- [YOLOv5](https://github.com/ultralytics/yolov5) (via `torch.hub`) — object detection
- [Norfair](https://github.com/tryolabs/norfair) — multi-object tracking & camera motion estimation
- [OpenCV](https://opencv.org/) — frame I/O and video handling
- [Pillow](https://python-pillow.org/) — overlay rendering
- [NumPy](https://numpy.org/) / [Pandas](https://pandas.pydata.org/) — data processing
- [Poetry](https://python-poetry.org/) — dependency management

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- Built on top of [Norfair](https://github.com/tryolabs/norfair), the lightweight real-time object tracking library by [Tryolabs](https://tryolabs.com/).
- Inspired by Tryolabs' soccer video analytics work.
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the detection backbone.
