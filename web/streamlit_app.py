"""Streamlit front-end for RF-DETR aquaculture monitoring.

Run:
    streamlit run web/streamlit_app.py

Reads STREAM_URL, RFDETR_MODEL_PATH, RFDETR_DEVICE, and
RFDETR_CONFIDENCE_THRESHOLD from .env (or environment).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import streamlit as st

# Load .env before importing project modules that read settings
from config.setting import (
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

# Video placeholder
video_placeholder = st.empty()

# Status placeholder
status_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Display loop (runs while the pipeline is active)
# ---------------------------------------------------------------------------

if st.session_state.running and st.session_state.pipeline:
    pipeline: InferencePipeline = st.session_state.pipeline
    status_placeholder.info("Waiting for first frame…")

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
