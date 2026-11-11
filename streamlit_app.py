"""Streamlit UI for the soccer video analytics pipeline.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from football_analytics.config import (
    BALL_MODEL_PATH,
    DEFAULT_PRESET,
    OUTPUTS_DIR,
    PLAYER_MODEL_PATH,
    PRESETS,
    VIDEOS_DIR,
    Settings,
    device_summary,
    resolve_device,
)
from football_analytics.detection import Detector
from football_analytics.pipeline import analyze_video, export_results

st.set_page_config(page_title="Soccer Video Analytics", page_icon="⚽", layout="wide")

TEAM_NAMES = ["Team A", "Team B"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]


@st.cache_resource(show_spinner="Loading models...")
def load_detector(player_model: str, ball_model: str | None, device: str, half: bool) -> Detector:
    """Loaded once per model/device combination and reused across reruns."""
    return Detector(player_model=player_model, ball_model=ball_model, device=device, half=half)


def sidebar() -> tuple[Settings, str]:
    st.sidebar.header("Settings")
    st.sidebar.caption(f"Device: {device_summary()}")
    if not PLAYER_MODEL_PATH.exists():
        st.sidebar.warning(
            "Football models not found, using the generic YOLO fallback "
            "(no goalkeeper/referee classes, weaker ball detection). Run "
            "`python scripts/download_models.py` once to fix this."
        )

    preset_name = st.sidebar.radio(
        "Speed preset", list(PRESETS), index=list(PRESETS).index(DEFAULT_PRESET), horizontal=True
    )
    preset = PRESETS[preset_name]
    defaults = Settings()

    draw_possession = st.sidebar.toggle("Possession", value=True)
    draw_passes = st.sidebar.toggle("Passes", value=True)
    use_ball_model = st.sidebar.toggle(
        "Dedicated ball model (slower, more accurate)",
        value=False,
        disabled=not BALL_MODEL_PATH.exists(),
    )

    with st.sidebar.expander("Advanced"):
        imgsz = st.select_slider("Inference size", [640, 960, 1280], value=preset["imgsz"])
        frame_stride = st.slider("Detect every Nth frame", 1, 5, preset["frame_stride"])
        batch_size = st.slider("Batch size", 1, 8, preset["batch_size"])
        player_conf = st.slider("Player confidence", 0.1, 0.9, defaults.player_conf, 0.05)
        ball_conf = st.slider("Ball confidence", 0.05, 0.9, defaults.ball_conf, 0.05)
        possession_ratio = st.slider(
            "Possession distance (x player height)", 0.2, 1.5, defaults.possession_distance_ratio, 0.05
        )
        half = st.checkbox("FP16 (GPU only)", value=defaults.half)
        player_model = st.text_input("Player model", value=defaults.player_model)
        encoder = st.selectbox("Video encoder", ["libx264", "h264_nvenc"])
        max_frames = st.number_input("Max frames (0 = whole video)", min_value=0, value=0, step=100)

    settings = defaults.with_overrides(
        imgsz=imgsz,
        frame_stride=frame_stride,
        batch_size=batch_size,
        player_conf=player_conf,
        ball_conf=ball_conf,
        possession_distance_ratio=possession_ratio,
        half=half,
        player_model=player_model,
        use_ball_model=use_ball_model,
        draw_possession=draw_possession,
        draw_passes=draw_passes,
        device=resolve_device("auto"),
    )
    st.session_state["encoder"] = encoder
    st.session_state["max_frames"] = int(max_frames) or None
    return settings, preset_name


def choose_video() -> Path | None:
    upload_tab, sample_tab = st.tabs(["Upload a video", "Use a local video"])
    with upload_tab:
        uploaded = st.file_uploader("Match video", type=VIDEO_TYPES)
        if uploaded is not None:
            digest = hashlib.sha1(uploaded.getbuffer()).hexdigest()[:12]
            path = Path(tempfile.gettempdir()) / f"sva_{digest}{Path(uploaded.name).suffix}"
            if not path.exists():
                path.write_bytes(uploaded.getbuffer())
            return path
    with sample_tab:
        local = sorted(p for ext in VIDEO_TYPES for p in VIDEOS_DIR.glob(f"*.{ext}"))
        if not local:
            st.info(
                "No videos in `videos/`. Run `python scripts/download_models.py --videos` "
                "to get five sample clips."
            )
            return None
        choice = st.selectbox("Video", local, format_func=lambda p: p.name)
        if st.toggle("Use this video", value=False):
            return choice
    return None


def run_analysis(video: Path, settings: Settings) -> None:
    detector = load_detector(
        settings.player_model,
        str(BALL_MODEL_PATH) if settings.use_ball_model else None,
        settings.device,
        settings.half,
    )
    out_dir = OUTPUTS_DIR / video.stem
    bar = st.progress(0.0, text="Calibrating team colours...")

    def progress(done: int, total: int, fps: float) -> None:
        remaining = (total - done) / fps if fps else 0
        bar.progress(
            min(done / total, 1.0) if total else 0.0,
            text=f"Frame {done}/{total} · {fps:.1f} fps · ~{remaining:.0f}s left",
        )

    result = analyze_video(
        video,
        out_dir / "annotated.mp4",
        settings,
        detector=detector,
        progress=progress,
        max_frames=st.session_state.get("max_frames"),
        encoder=st.session_state.get("encoder", "libx264"),
    )
    export_results(result, out_dir)
    bar.empty()
    st.session_state["result"] = result


def show_results() -> None:
    result = st.session_state.get("result")
    if result is None:
        return
    summary = result.summary
    colors = summary["team_colors"]

    if summary.get("fallback_model"):
        st.warning("Processed with the generic fallback model; download the football models for better accuracy.")
    if not summary.get("teams_detected"):
        st.warning("Could not find enough players to separate the two teams.")

    st.success(
        f"Processed {summary['frames']} frames in {summary['processing_s']:.0f}s "
        f"({summary['processing_fps']:.1f} fps on {summary['device']})."
    )

    video_col, stats_col = st.columns([3, 2])
    with video_col:
        st.video(str(result.output_video))
    with stats_col:
        cols = st.columns(2)
        for team, col in enumerate(cols):
            col.markdown(
                f"<div style='display:flex;align-items:center;gap:8px'>"
                f"<span style='width:16px;height:16px;border:1px solid #888;background:{colors[team]};display:inline-block'></span>"
                f"<b>{TEAM_NAMES[team]}</b></div>",
                unsafe_allow_html=True,
            )
            col.metric("Possession", f"{summary['possession_pct'][team]:.0f}%")
            col.metric("Passes", summary["passes"][team])
            col.metric("Turnovers won", summary["turnovers_won"][team])

        fig = go.Figure(
            go.Bar(
                x=summary["possession_pct"],
                y=["Possession"] * 2,
                orientation="h",
                marker_color=colors,
                marker_line_color="#888",
                marker_line_width=1,
                text=[f"{TEAM_NAMES[t]} {p:.0f}%" for t, p in enumerate(summary["possession_pct"])],
                textposition="inside",
            )
        )
        fig.update_layout(barmode="stack", height=120, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        fig.update_xaxes(range=[0, 100], visible=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True)

    timeline = result.timeline_frame()
    if not timeline.empty:
        timeline = timeline[timeline["team"] >= 0].copy()
    if not timeline.empty:
        st.subheader("Possession over time")
        timeline["Team"] = timeline["team"].map(dict(enumerate(TEAM_NAMES)))
        window = max(1, int(summary["video_fps"] * 5 / max(1, summary["settings"]["frame_stride"])))
        for team, name in enumerate(TEAM_NAMES):
            timeline[name] = (timeline["team"] == team).rolling(window, min_periods=1).mean() * 100
        fig = px.area(
            timeline,
            x="time_s",
            y=TEAM_NAMES,
            color_discrete_sequence=colors,
            labels={"time_s": "Time (s)", "value": "Possession (%, 5 s window)", "variable": ""},
        )
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    events = result.events_frame()
    st.subheader("Passes and turnovers")
    if events.empty:
        st.caption("No passes detected.")
    else:
        events = events.assign(team=events["team"].map(dict(enumerate(TEAM_NAMES))))
        st.dataframe(events, use_container_width=True, hide_index=True)

    st.subheader("Downloads")
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Annotated video (MP4)",
        result.output_video.read_bytes(),
        file_name=f"{result.output_video.parent.name}_annotated.mp4",
        mime="video/mp4",
    )
    d2.download_button(
        "Events (CSV)",
        result.events_frame().to_csv(index=False),
        file_name="events.csv",
        mime="text/csv",
    )
    d3.download_button(
        "Summary (JSON)",
        json.dumps(summary, indent=2),
        file_name="summary.json",
        mime="application/json",
    )


def main() -> None:
    st.title("⚽ Soccer Video Analytics")
    st.caption(
        "Upload a broadcast-style match clip to get team detection, ball tracking, "
        "possession and passes. Pretrained models, no training needed."
    )
    settings, _ = sidebar()
    video = choose_video()
    if video is not None and st.button("Analyze", type="primary"):
        with st.spinner("Analyzing..."):
            run_analysis(video, settings)
    show_results()


main()
