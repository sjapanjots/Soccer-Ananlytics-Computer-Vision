import os
import tempfile

import cv2
import numpy as np
import PIL
import PIL.ImageDraw
import streamlit as st
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean

from inference import Converter, HSVClassifier, InertiaClassifier, YoloV5
from inference.colors import all as all_colors
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator,
)
from soccer import Match, Player, Team
from soccer.draw import AbsolutePath
from soccer.pass_event import Pass

st.set_page_config(
    page_title="Soccer Analytics",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Soccer Analytics — Computer Vision")
st.markdown(
    "AI-powered soccer match analysis. Upload a video, configure teams and settings, "
    "then process to detect players, track ball possession, and detect passes."
)

# ── Sidebar: Configuration ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    st.subheader("Video")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    st.subheader("Ball Detection Model")
    model_path = st.text_input(
        "Path to custom ball model (.pt)",
        value="models/ball.pt",
    )

    st.subheader("Home Team")
    home_name = st.text_input("Team Name", value="Chelsea", key="home_name")
    home_abbr = st.text_input("Abbreviation (3 chars)", value="CHE", key="home_abbr")
    home_color = st.color_picker("Team Color", value="#FF0000", key="home_color")
    home_board_color = st.color_picker("Board Color", value="#F45640", key="home_board")
    home_text_color = st.color_picker("Text Color", value="#FFFFFF", key="home_text")

    st.subheader("Away Team")
    away_name = st.text_input("Team Name", value="Man City", key="away_name")
    away_abbr = st.text_input("Abbreviation (3 chars)", value="MNC", key="away_abbr")
    away_color = st.color_picker("Team Color", value="#F0E6BC", key="away_color")

    # ── HSV Filters ──────────────────────────────────────────────────────
    st.subheader("HSV Jersey Filters")

    with st.expander("Home Team Kit Colors", expanded=True):
        home_color_names = st.multiselect(
            "Select jersey colors for the home team",
            options=[c["name"] for c in all_colors],
            default=["blue", "green"],
            key="home_color_names",
        )

    with st.expander("Away Team Kit Colors", expanded=True):
        away_color_names = st.multiselect(
            "Select jersey colors for the away team",
            options=[c["name"] for c in all_colors],
            default=["sky_blue"],
            key="away_color_names",
        )

    with st.expander("Referee Kit Colors", expanded=True):
        ref_color_names = st.multiselect(
            "Select jersey colors for the referee",
            options=[c["name"] for c in all_colors],
            default=["black"],
            key="ref_color_names",
        )

    with st.expander("Custom HSV Colors", expanded=False):
        st.caption("Optionally add one or more custom color ranges (e.g. special kit).")
        n_custom = st.number_input("Number of custom colors", 0, 10, 0, key="n_custom")
        custom_colors = []
        for i in range(int(n_custom)):
            st.markdown(f"**Custom Color {i+1}**")
            c_name = st.text_input(f"Custom color {i+1} name", value=f"custom_{i+1}", key=f"cn{i}")
            c_hmin = st.number_input(f"{c_name}: Lower H", 0, 179, 0, key=f"chmin{i}")
            c_smin = st.number_input(f"{c_name}: Lower S", 0, 255, 0, key=f"csmin{i}")
            c_vmin = st.number_input(f"{c_name}: Lower V", 0, 255, 0, key=f"cvmin{i}")
            c_hmax = st.number_input(f"{c_name}: Upper H", 0, 179, 179, key=f"chmax{i}")
            c_smax = st.number_input(f"{c_name}: Upper S", 0, 255, 255, key=f"csmax{i}")
            c_vmax = st.number_input(f"{c_name}: Upper V", 0, 255, 255, key=f"cvmax{i}")
            custom_colors.append({
                "name": c_name,
                "lower_hsv": (int(c_hmin), int(c_smin), int(c_vmin)),
                "upper_hsv": (int(c_hmax), int(c_smax), int(c_vmax)),
            })

    # ── Tracker Settings ─────────────────────────────────────────────────
    st.subheader("Player Tracker")
    player_dist_threshold = st.slider("Distance threshold", 50, 500, 250, key="pdt")
    player_init_delay = st.slider("Initialization delay", 0, 20, 3, key="pid")
    player_hit_max = st.slider("Hit counter max", 10, 500, 90, key="phm")

    st.subheader("Ball Tracker")
    ball_dist_threshold = st.slider(
        "Ball distance threshold", 50, 500, 150, key="bdt"
    )
    ball_init_delay = st.slider("Ball initialization delay", 0, 50, 20, key="bid")
    ball_hit_max = st.slider("Ball hit counter max", 100, 5000, 2000, key="bhm")

    # ── Match Settings ───────────────────────────────────────────────────
    st.subheader("Match Settings")
    possession_threshold = st.slider(
        "Possession change threshold (frames)", 5, 60, 20, key="pct"
    )
    ball_distance_threshold = st.slider(
        "Ball distance threshold (px)", 10, 200, 45, key="bdt2"
    )
    inertia_value = st.slider(
        "Classification inertia", 1, 100, 20, key="inertia"
    )

    # ── Features ─────────────────────────────────────────────────────────
    st.subheader("Features")
    enable_possession = st.checkbox("Possession Counter", value=True)
    enable_passes = st.checkbox("Pass Detection", value=True)
    enable_ids = st.checkbox("Show Player IDs", value=True)


# ── Helper: Convert hex color to RGB tuple ──────────────────────────────────
def hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


# ── Helper: Ensure counter board images exist ───────────────────────────────
def ensure_board_images() -> bool:
    """Create missing possession/pass counter boards so match drawing works.

    The board PNGs are gitignored, so on a fresh checkout they may be absent.
    If they are missing, generate simple semi-transparent placeholders.
    """
    boards = [
        (os.path.join("images", "possession_board.png"), "POSSESSION"),
        (os.path.join("images", "passes_board.png"), "PASSES"),
    ]

    missing = [path for path, _ in boards if not os.path.exists(path)]
    if not missing:
        return True

    os.makedirs("images", exist_ok=True)

    for path, label in boards:
        if os.path.exists(path):
            continue
        img = PIL.Image.new("RGBA", (315, 210), (30, 30, 30, 255))
        draw = PIL.ImageDraw.Draw(img)
        draw.rectangle([3, 3, 311, 206], outline=(255, 255, 255, 120), width=2)
        draw.text((315 / 2 - 50, 210 / 2 - 15), label, fill=(255, 255, 255, 255))
        img.save(path)

    return True


# ── Helper: Resolve selected color names to filter structs ──────────────────
def colors_from_names(names: list) -> list:
    return [c for c in all_colors if c["name"] in names]


def build_filters(
    home_name: str,
    away_name: str,
    home_color_names: list,
    away_color_names: list,
    ref_color_names: list,
    custom_colors: list,
) -> list:
    home_colors = colors_from_names(home_color_names)
    away_colors = colors_from_names(away_color_names)
    ref_colors = colors_from_names(ref_color_names)

    if not home_colors:
        raise ValueError(f"Select at least one jersey color for the home team ({home_name}).")
    if not away_colors:
        raise ValueError(f"Select at least one jersey color for the away team ({away_name}).")
    if not ref_colors:
        raise ValueError("Select at least one jersey color for the referee.")

    filters = [
        {"name": home_name, "colors": home_colors},
        {"name": away_name, "colors": away_colors},
        {"name": "Referee", "colors": ref_colors},
    ]

    # Only include non-empty custom colors
    valid_custom = [c for c in custom_colors if c is not None]
    if valid_custom:
        filters.append({"name": "Custom", "colors": valid_custom})

    return filters


# ── Processing Function ─────────────────────────────────────────────────────
def process_video(
    video_path: str,
    model_path: str,
    output_path: str,
    home_team: Team,
    away_team: Team,
    filters: list,
    player_tracker_cfg: dict,
    ball_tracker_cfg: dict,
    possession_threshold: int,
    ball_distance_threshold: int,
    inertia_value: int,
    enable_possession: bool,
    enable_passes: bool,
    enable_ids: bool,
    progress_bar=None,
    status_text=None,
):
    teams = [home_team, away_team]

    video = Video(input_path=video_path, output_path=output_path)
    fps = video.video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Detectors
    player_detector = YoloV5()
    ball_detector = YoloV5(model_path=model_path)

    # Classifier
    hsv_classifier = HSVClassifier(filters=filters)
    classifier = InertiaClassifier(classifier=hsv_classifier, inertia=inertia_value)

    # Match
    match = Match(home=home_team, away=away_team, fps=fps)
    match.possession_counter_threshold = possession_threshold
    match.ball_distance_threshold = ball_distance_threshold

    # Trackers
    player_tracker = Tracker(
        distance_function=mean_euclidean,
        distance_threshold=player_tracker_cfg["distance_threshold"],
        initialization_delay=player_tracker_cfg["initialization_delay"],
        hit_counter_max=player_tracker_cfg["hit_counter_max"],
    )
    ball_tracker = Tracker(
        distance_function=mean_euclidean,
        distance_threshold=ball_tracker_cfg["distance_threshold"],
        initialization_delay=ball_tracker_cfg["initialization_delay"],
        hit_counter_max=ball_tracker_cfg["hit_counter_max"],
    )
    motion_estimator = MotionEstimator()
    coord_transformations = None

    # Paths
    path = AbsolutePath()

    # Backgrounds
    possession_background = match.get_possession_background()
    passes_background = match.get_passes_background()

    for i, frame in enumerate(video):
        if progress_bar and total_frames > 0:
            progress_bar.progress(min((i + 1) / total_frames, 1.0))
        if status_text:
            status_text.text(f"Processing frame {i + 1} / {total_frames}...")

        # Detections
        players_detections = get_player_detections(player_detector, frame)
        ball_detections = get_ball_detections(ball_detector, frame)
        detections = ball_detections + players_detections

        # Motion
        coord_transformations = update_motion_estimator(
            motion_estimator=motion_estimator,
            detections=detections,
            frame=frame,
        )

        # Tracking
        player_track_objects = player_tracker.update(
            detections=players_detections, coord_transformations=coord_transformations
        )
        ball_track_objects = ball_tracker.update(
            detections=ball_detections, coord_transformations=coord_transformations
        )

        player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
        ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)

        # Classification
        player_detections = classifier.predict_from_detections(
            detections=player_detections, img=frame
        )

        # Match update
        ball = get_main_ball(ball_detections, match)
        players = Player.from_detections(detections=player_detections, teams=teams)
        match.update(players, ball)

        # Draw
        frame = PIL.Image.fromarray(frame)

        if enable_possession:
            frame = Player.draw_players(
                players=players, frame=frame, confidence=False, id=enable_ids
            )
            if coord_transformations is not None:
                frame = path.draw(
                    img=frame,
                    detection=ball.detection,
                    coord_transformations=coord_transformations,
                    color=match.team_possession.color,
                )
            frame = match.draw_possession_counter(
                frame, counter_background=possession_background, debug=False
            )
            if ball:
                frame = ball.draw(frame)

        if enable_passes:
            pass_list = match.passes
            if coord_transformations is not None:
                frame = Pass.draw_pass_list(
                    img=frame,
                    passes=pass_list,
                    coord_transformations=coord_transformations,
                )
            frame = match.draw_passes_counter(
                frame, counter_background=passes_background, debug=False
            )

        frame = np.array(frame)
        video.write(frame)

    if progress_bar:
        progress_bar.progress(1.0)
    if status_text:
        status_text.text("Processing complete!")

    return match


# ── Main Content ────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Video")
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        video_path = tfile.name
        st.video(video_file)
        st.success(f"Video loaded: {video_file.name}")

        cap = cv2.VideoCapture(video_path)
        vfps = cap.get(cv2.CAP_PROP_FPS)
        vframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vwidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        st.info(
            f"Resolution: {vwidth}x{vheight} | FPS: {vfps:.1f} | "
            f"Frames: {vframes} | Duration: {vframes / vfps:.1f}s"
        )
    else:
        st.info("Please upload a video file to get started.")

with col2:
    st.header("Results")
    results_placeholder = st.empty()

# ── Process Button ──────────────────────────────────────────────────────────
st.divider()

if st.button("🚀 Process Video", type="primary", use_container_width=True):
    if video_file is None:
        st.error("Please upload a video first!")
    else:
        # Build teams
        try:
            home_team = Team(
                name=home_name,
                abbreviation=home_abbr.upper(),
                color=hex_to_rgb(home_color),
                board_color=hex_to_rgb(home_board_color),
                text_color=hex_to_rgb(home_text_color),
            )
            away_team = Team(
                name=away_name,
                abbreviation=away_abbr.upper(),
                color=hex_to_rgb(away_color),
            )
        except ValueError as e:
            st.error(f"Team configuration error: {e}")
            st.stop()

        # Build filters
        try:
            filters_list = build_filters(
                home_name=home_name,
                away_name=away_name,
                home_color_names=home_color_names,
                away_color_names=away_color_names,
                ref_color_names=ref_color_names,
                custom_colors=custom_colors,
            )
        except ValueError as e:
            st.error(f"HSV filter configuration error: {e}")
            st.stop()

        # Check if model exists
        if not os.path.exists(model_path):
            st.error(
                f"Ball model not found at `{model_path}`. "
                "Please provide a valid path to your custom YOLOv5 ball detection model (.pt)."
            )
            st.stop()

        # Ensure counter board images exist
        ensure_board_images()

        # Process
        output_path = tempfile.mktemp(suffix="_output.mp4")

        player_tracker_cfg = {
            "distance_threshold": player_dist_threshold,
            "initialization_delay": player_init_delay,
            "hit_counter_max": player_hit_max,
        }
        ball_tracker_cfg = {
            "distance_threshold": ball_dist_threshold,
            "initialization_delay": ball_init_delay,
            "hit_counter_max": ball_hit_max,
        }

        progress_bar = st.progress(0, text="Starting processing...")
        status_text = st.empty()

        match = process_video(
            video_path=video_path,
            model_path=model_path,
            output_path=output_path,
            home_team=home_team,
            away_team=away_team,
            filters=filters_list,
            player_tracker_cfg=player_tracker_cfg,
            ball_tracker_cfg=ball_tracker_cfg,
            possession_threshold=possession_threshold,
            ball_distance_threshold=ball_distance_threshold,
            inertia_value=inertia_value,
            enable_possession=enable_possession,
            enable_passes=enable_passes,
            enable_ids=enable_ids,
            progress_bar=progress_bar,
            status_text=status_text,
        )

        # Show results
        with results_placeholder.container():
            st.subheader("Processed Video")
            if os.path.exists(output_path):
                st.video(output_path)

                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f.read(),
                        file_name="soccer_analytics_output.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
            else:
                st.error("Output video was not generated.")

            # Match Statistics
            st.subheader("Match Statistics")

            stat_col1, stat_col2 = st.columns(2)

            with stat_col1:
                st.markdown(f"**{match.home.name}** ({match.home.abbreviation})")
                st.metric(
                    "Possession Time",
                    match.home.get_time_possession(match.fps),
                )
                st.metric("Passes", len(match.home.passes))

            with stat_col2:
                st.markdown(f"**{match.away.name}** ({match.away.abbreviation})")
                st.metric(
                    "Possession Time",
                    match.away.get_time_possession(match.fps),
                )
                st.metric("Passes", len(match.away.passes))

            home_pct = match.home.get_percentage_possession(match.duration) * 100
            away_pct = match.away.get_percentage_possession(match.duration) * 100
            st.progress(home_pct / 100)
            st.caption(
                f"{match.home.abbreviation}: {home_pct:.1f}% | "
                f"{match.away.abbreviation}: {away_pct:.1f}%"
            )

        # Cleanup
        try:
            os.remove(output_path)
        except OSError:
            pass

# ── Quick Reference: Available HSV Color Definitions ────────────────────────
st.divider()
with st.expander("📖 Quick Reference: Available HSV Color Definitions"):
    for color in all_colors:
        st.markdown(
            f"- **{color['name']}**: lower_hsv={color['lower_hsv']}, "
            f"upper_hsv={color['upper_hsv']}"
        )