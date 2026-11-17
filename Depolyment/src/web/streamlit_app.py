"""Streamlit front-end for RF-DETR aquaculture monitoring.

Run:
    streamlit run Depolyment/src/web/streamlit_app.py

Reads STREAM_URL, RFDETR_MODEL_PATH, RFDETR_DEVICE, and
RFDETR_CONFIDENCE_THRESHOLD from config.setting (edit that file, not .env).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import plotly.graph_objects as go
import requests
import streamlit as st

# Load .env before importing project modules that read settings
from config.setting import (
    PI_CONTROLLER_URL,
    RFDETR_CONFIDENCE_THRESHOLD,
    RFDETR_DEVICE,
    RFDETR_MODEL_PATH,
    STREAM_URL,
)
from web.pipeline import InferencePipeline, draw_detections
from web.trt import detect_support

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RF-DETR Aquaculture Monitor",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

_TARGET_DISPLAY_FPS = 30
_DISPLAY_INTERVAL = 1.0 / _TARGET_DISPLAY_FPS
_CHART_UPDATE_INTERVAL = 0.5  # seconds between chart refreshes

_HUNGER_COLORS = {"Mild": "#2ecc71", "Moderate": "#f39c12", "Severe": "#e74c3c"}
_HUNGER_LABELS = {
    "Mild": "Mild",
    "Moderate": "Moderate",
    "Severe": "Severe",
}
_HUNGER_SEVERITY = {"Mild": 0, "Moderate": 1, "Severe": 2}
_FEED_TRIGGER_LEVELS = {"Moderate", "Severe"}
_FEED_ASSESSMENT_WINDOW_SECONDS = 30.0


def _trigger_manual_feed(pi_url: str, hunger_level: str | None = None) -> tuple[bool, str]:
    """POST /feed to the Raspberry Pi controller. Returns (success, message)."""
    url = pi_url.rstrip("/") + "/feed"
    payload = None
    if hunger_level is not None:
        payload = {
            "hunger_level": hunger_level,
            "window_seconds": _FEED_ASSESSMENT_WINDOW_SECONDS,
        }
    try:
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            bmduino = data.get("bmduino", [])
            suffix = f" BMduino: {' | '.join(bmduino)}" if bmduino else ""
            if hunger_level is None:
                return True, f"Feed signal sent.{suffix}"
            label = _HUNGER_LABELS.get(hunger_level, hunger_level)
            return True, f"Additional feed triggered. Highest hunger level: {label}.{suffix}"
        data = resp.json()
        return False, data.get("detail", {}).get("message", resp.text) if isinstance(data.get("detail"), dict) else str(data.get("detail", resp.text))
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to Pi ({pi_url})"
    except requests.exceptions.Timeout:
        return False, "Connection timeout"
    except Exception as e:
        return False, str(e)


def _build_area_chart(data: dict) -> go.Figure:
    """Build the feed area chart with Raw / Kalman / EMA lines."""
    frame_ids = data["frame_ids"]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frame_ids, y=data["raw"],
        name="Raw (RF-DETR)", mode="lines",
        line=dict(color="rgba(180,40,40,0.45)", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=frame_ids, y=data["kalman"],
        name="Kalman Filter", mode="lines",
        line=dict(color="#3498db", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=frame_ids, y=data["ema"],
        name="EMA", mode="lines",
        line=dict(color="#e67e22", width=2.5),
    ))

    fig.update_layout(
        title="Feed Area / Frame",
        xaxis_title="Frame",
        yaxis_title="Feed Area (px²)",
        yaxis=dict(tickformat=".0f"),
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=50, r=20, t=50, b=40),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig


def _more_severe_level(current: str | None, candidate: str) -> str:
    """Return the more severe hunger level."""
    if current is None:
        return candidate
    if _HUNGER_SEVERITY.get(candidate, -1) > _HUNGER_SEVERITY.get(current, -1):
        return candidate
    return current


def _ensure_feed_assessment_state() -> None:
    if "feed_assessment_active" not in st.session_state:
        st.session_state.feed_assessment_active = False
    if "feed_assessment_started_at" not in st.session_state:
        st.session_state.feed_assessment_started_at = 0.0
    if "feed_assessment_ends_at" not in st.session_state:
        st.session_state.feed_assessment_ends_at = 0.0
    if "feed_assessment_worst_level" not in st.session_state:
        st.session_state.feed_assessment_worst_level = None
    if "feed_status_message" not in st.session_state:
        st.session_state.feed_status_message = ""
    if "feed_status_kind" not in st.session_state:
        st.session_state.feed_status_kind = "info"


def _start_feed_assessment_window() -> None:
    now = time.monotonic()
    st.session_state.feed_assessment_active = True
    st.session_state.feed_assessment_started_at = now
    st.session_state.feed_assessment_ends_at = now + _FEED_ASSESSMENT_WINDOW_SECONDS
    st.session_state.feed_assessment_worst_level = None
    st.session_state.feed_status_kind = "info"
    st.session_state.feed_status_message = "Assessment window started. Additional feed will trigger if Moderate/Severe hunger detected in next 30s."


def _render_feed_status(feed_status) -> None:
    if "feed_status_message" not in st.session_state or "feed_status_kind" not in st.session_state:
        return
    message = st.session_state.feed_status_message
    if not message:
        return
    kind = st.session_state.feed_status_kind
    if kind == "success":
        feed_status.success(message)
    elif kind == "error":
        feed_status.error(message)
    elif kind == "warning":
        feed_status.warning(message)
    else:
        feed_status.info(message)


def _on_feed_button_click(pi_url: str) -> None:
    """Callback for Feed button click. Avoids full page rerun."""
    if not st.session_state.get("running", False) or st.session_state.pipeline is None:
        st.session_state.feed_status_kind = "warning"
        st.session_state.feed_status_message = "Press Start first to begin detection before using hunger level assessment."
    elif st.session_state.feed_assessment_active:
        remaining = max(0.0, st.session_state.feed_assessment_ends_at - time.monotonic())
        st.session_state.feed_status_kind = "warning"
        st.session_state.feed_status_message = f"30s assessment window in progress, {remaining:.0f}s remaining."
    else:
        ok, msg = _trigger_manual_feed(pi_url)
        if ok:
            _start_feed_assessment_window()
        else:
            st.session_state.feed_status_kind = "error"
            st.session_state.feed_status_message = msg


def _update_feed_assessment(snapshot, pi_url: str) -> None:
    if not st.session_state.feed_assessment_active:
        return

    if snapshot is not None:
        st.session_state.feed_assessment_worst_level = _more_severe_level(
            st.session_state.feed_assessment_worst_level,
            snapshot.hunger_level,
        )

    now = time.monotonic()
    worst_level = st.session_state.feed_assessment_worst_level
    worst_label = _HUNGER_LABELS.get(worst_level, "No data")

    if now < st.session_state.feed_assessment_ends_at:
        remaining = st.session_state.feed_assessment_ends_at - now
        st.session_state.feed_status_kind = "info"
        st.session_state.feed_status_message = (
            f"Assessment in progress: {remaining:.0f}s remaining. Highest hunger level: {worst_label}."
        )
        return

    st.session_state.feed_assessment_active = False
    st.session_state.feed_assessment_started_at = 0.0
    st.session_state.feed_assessment_ends_at = 0.0
    st.session_state.feed_assessment_worst_level = None

    if worst_level in _FEED_TRIGGER_LEVELS:
        ok, msg = _trigger_manual_feed(pi_url, worst_level)
        st.session_state.feed_status_kind = "success" if ok else "error"
        st.session_state.feed_status_message = msg
    else:
        st.session_state.feed_status_kind = "info"
        st.session_state.feed_status_message = (
            f"30s assessment complete. Highest hunger level: {worst_label}. Below Moderate/Severe threshold, no additional feed triggered."
        )


def _build_rate_chart(data: dict) -> go.Figure:
    """Build the consumption rate chart with hunger zone bands."""
    frame_ids = data["frame_ids"]
    rates = data["consumption_rate"]

    fig = go.Figure()

    # Hunger zone background bands
    fig.add_hrect(y0=-0.05, y1=0.5,  fillcolor="rgba(46,204,113,0.12)", line_width=0, annotation_text="Mild",     annotation_position="top right")
    fig.add_hrect(y0=-0.10, y1=-0.05, fillcolor="rgba(243,156,18,0.15)", line_width=0, annotation_text="Moderate", annotation_position="right")
    fig.add_hrect(y0=-0.5,  y1=-0.10, fillcolor="rgba(231,76,60,0.15)",  line_width=0, annotation_text="Severe",   annotation_position="right")

    # Threshold lines
    fig.add_hline(y=-0.05, line_dash="dash", line_color="rgba(46,204,113,0.7)",  line_width=1.5)
    fig.add_hline(y=-0.10, line_dash="dash", line_color="rgba(231,76,60,0.7)",   line_width=1.5)

    # Consumption rate line
    fig.add_trace(go.Scatter(
        x=frame_ids, y=rates,
        name="Consumption rate (windowed)", mode="lines",
        line=dict(color="#e74c3c", width=2),
    ))

    fig.update_layout(
        title="Consumption Rate & Hunger Zones",
        xaxis_title="Frame",
        yaxis_title="Rate (log/frame)",
        yaxis=dict(range=[-0.35, 0.15]),
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=50, r=20, t=50, b=40),
        height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig

# ---------------------------------------------------------------------------
# Sidebar – controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("RF-DETR Monitor")
    st.divider()

    stream_url = st.text_input("RTSP stream URL", value=STREAM_URL)

    model_path_str = st.text_input(
        "Model path (.pth)",
        value=RFDETR_MODEL_PATH or "models/rf-detr-fish-regular.pth",
    )

    device_options = ["auto", "cuda", "cpu", "mps"]
    device = st.selectbox(
        "Device",
        options=device_options,
        index=device_options.index(RFDETR_DEVICE) if RFDETR_DEVICE in device_options else 0,
    )

    conf = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(RFDETR_CONFIDENCE_THRESHOLD),
        step=0.05,
    )

    trt_support = detect_support()
    use_trt = st.toggle(
        "TensorRT acceleration",
        value=trt_support["tensorrt"],
        help=(
            "Uses torch_tensorrt backend when available. "
            "Falls back to torch.compile (inductor) on CUDA, "
            "or no compilation on CPU/MPS."
        ),
    )
    if trt_support["tensorrt"]:
        st.caption("torch_tensorrt detected")
    elif trt_support["cuda"]:
        st.caption(f"CUDA available — torch.compile (inductor) will be used  \n_{trt_support['reason']}_")
    else:
        st.caption(f"No GPU — running on CPU  \n_{trt_support['reason']}_")

    st.divider()
    show_labels = st.checkbox("Show class labels", value=True)
    show_conf_score = st.checkbox("Show confidence score", value=True)

    st.divider()
    st.subheader("Feed Control")
    pi_url = st.text_input("Pi Controller URL", value=PI_CONTROLLER_URL)
    st.button(
        "🐟 Manual Feed",
        use_container_width=True,
        type="primary",
        on_click=_on_feed_button_click,
        args=(pi_url,),
    )
    feed_status = st.empty()
    _ensure_feed_assessment_state()
    _render_feed_status(feed_status)

    st.divider()
    col_start, col_stop = st.columns(2)
    start_clicked = col_start.button("▶ Start", use_container_width=True, type="primary")
    stop_clicked = col_stop.button("■ Stop", use_container_width=True)



# ---------------------------------------------------------------------------
# Session state – pipeline lifecycle
# ---------------------------------------------------------------------------

if "pipeline" not in st.session_state:
    st.session_state.pipeline: InferencePipeline | None = None
if "running" not in st.session_state:
    st.session_state.running = False
if "display_fps_counter" not in st.session_state:
    st.session_state.display_fps_counter = 0
if "display_fps" not in st.session_state:
    st.session_state.display_fps = 0.0
if "display_fps_t0" not in st.session_state:
    st.session_state.display_fps_t0 = time.monotonic()
if "chart_t0" not in st.session_state:
    st.session_state.chart_t0 = time.monotonic()
if "chart_gen" not in st.session_state:
    st.session_state.chart_gen = 0

# Handle start
if start_clicked and not st.session_state.running:
    model_path = Path(model_path_str)
    if not model_path.exists():
        st.sidebar.error(f"Model not found: {model_path}")
    elif not stream_url.strip():
        st.sidebar.error("Stream URL is required")
    else:
        with st.spinner("Loading RF-DETR model…"):
            pipeline = InferencePipeline(
                stream_url=stream_url,
                model_path=model_path,
                device=device,
                conf_threshold=conf,
                use_trt=use_trt,
            )
            pipeline.start()
        st.session_state.pipeline = pipeline
        st.session_state.running = True
        st.session_state.display_fps_t0 = time.monotonic()

# Handle stop
if stop_clicked and st.session_state.running:
    if st.session_state.pipeline:
        st.session_state.pipeline.stop()
        st.session_state.pipeline = None
    st.session_state.running = False
    if st.session_state.feed_assessment_active:
        st.session_state.feed_assessment_active = False
        st.session_state.feed_assessment_started_at = 0.0
        st.session_state.feed_assessment_ends_at = 0.0
        st.session_state.feed_assessment_worst_level = None
        st.session_state.feed_status_kind = "warning"
        st.session_state.feed_status_message = "Detection stopped. Ongoing 30s hunger level assessment cancelled."
        _render_feed_status(feed_status)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown("## Live Detection Feed")

# Optimization badge
if st.session_state.running and st.session_state.pipeline:
    info = st.session_state.pipeline.opt_info
    if info:
        badge_map = {
            "tensorrt": "🟢 TensorRT",
            "inductor":  "🟡 torch.compile (inductor)",
            "none":      "🔵 No compilation",
        }
        badge = badge_map.get(info.backend, info.backend)
        st.caption(f"{badge} · dtype={info.dtype} · device={device}")

# Metric columns (placeholders updated in the loop)
col_ifps, col_dfps, col_ims, col_ndet = st.columns(4)
metric_infer_fps  = col_ifps.empty()
metric_display_fps = col_dfps.empty()
metric_infer_ms   = col_ims.empty()
metric_ndet       = col_ndet.empty()

# Left-right layout: video on left, charts on right
left_col, right_col = st.columns([1.2, 1], gap="small")

# Left column: video and status
with left_col:
    video_placeholder = st.empty()
    status_placeholder = st.empty()

# Right column: feed analysis charts
with right_col:
    st.markdown("### Feed Analysis")
    hunger_badge = st.empty()
    area_chart_placeholder = st.empty()
    rate_chart_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Display loop (runs while the pipeline is active)
# ---------------------------------------------------------------------------

if st.session_state.running and st.session_state.pipeline:
    pipeline: InferencePipeline = st.session_state.pipeline
    status_placeholder.info("Waiting for first frame...")

    while st.session_state.running:
        t0 = time.monotonic()

        frame_result = pipeline.get_latest()

        if frame_result is None:
            time.sleep(0.05)
            continue

        status_placeholder.empty()

        # Draw detections on frame
        annotated = draw_detections(
            frame_result.frame,
            frame_result.result,
            show_labels=show_labels,
            show_conf=show_conf_score,
        )

        # Convert BGR → RGB for st.image
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, use_container_width=True)

        # Update metrics
        det_count = len(frame_result.result.detections) if frame_result.result else 0
        metric_infer_fps.metric("Inference FPS", f"{frame_result.infer_fps:.1f}")
        metric_infer_ms.metric("Inference ms", f"{frame_result.inference_ms:.1f}")
        metric_ndet.metric("Detections", det_count)

        # Track display FPS
        st.session_state.display_fps_counter += 1
        now = time.monotonic()
        elapsed = now - st.session_state.display_fps_t0
        if elapsed >= 1.0:
            st.session_state.display_fps = st.session_state.display_fps_counter / elapsed
            st.session_state.display_fps_counter = 0
            st.session_state.display_fps_t0 = now
        metric_display_fps.metric("Display FPS", f"{st.session_state.display_fps:.1f}")

        # Update feed area charts at ~2 FPS to keep UI responsive
        chart_elapsed = now - st.session_state.chart_t0
        if chart_elapsed >= _CHART_UPDATE_INTERVAL:
            st.session_state.chart_t0 = now
            st.session_state.chart_gen += 1
            gen = st.session_state.chart_gen

            snapshot = pipeline.get_latest_snapshot()
            chart_data = pipeline.get_chart_data()

            if snapshot is not None:
                level = snapshot.hunger_level
                color = _HUNGER_COLORS.get(level, "#aaa")
                label = _HUNGER_LABELS.get(level, level)
                hunger_badge.markdown(
                    f"**Hunger Level:** "
                    f"<span style='color:{color};font-weight:bold;font-size:1.1em'>{label}</span>"
                    f"&nbsp;&nbsp;&nbsp;rate = {snapshot.consumption_rate:+.4f} log/frame",
                    unsafe_allow_html=True,
                )

            _update_feed_assessment(snapshot, pi_url)
            _render_feed_status(feed_status)

            if chart_data["frame_ids"]:
                area_chart_placeholder.plotly_chart(
                    _build_area_chart(chart_data),
                    use_container_width=True,
                    key=f"area_{gen}",
                )
                rate_chart_placeholder.plotly_chart(
                    _build_rate_chart(chart_data),
                    use_container_width=True,
                    key=f"rate_{gen}",
                )

        # Pace display to 30 FPS
        spent = time.monotonic() - t0
        sleep = _DISPLAY_INTERVAL - spent
        if sleep > 0:
            time.sleep(sleep)

else:
    video_placeholder.info("Press **▶ Start** in the sidebar to begin streaming.")
    metric_infer_fps.metric("Inference FPS", "—")
    metric_display_fps.metric("Display FPS", "—")
    metric_infer_ms.metric("Inference ms", "—")
    metric_ndet.metric("Detections", "—")
