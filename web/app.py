"""Low-latency ripple monitoring Streamlit interface."""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time
from uuid import uuid4

from streaming.latest_frame_reader import LatestFrameReader
from core.ripple_detector import RippleDetector
from core.models import StreamHealthState, MonitoringSession
from core.stream_health import StreamHealthMonitor
from config.setting import LOCAL_TRUSTED_NETWORK_ONLY, STREAM_URL


MONITORING_STATES = {
    StreamHealthState.ACTIVE,
    StreamHealthState.INTERRUPTED,
    StreamHealthState.DETECTION_UNAVAILABLE,
}


def initialize_session_state():
    """Initialize or retrieve session state."""
    if "session" not in st.session_state:
        st.session_state.session = MonitoringSession(
            session_id=str(uuid4()),
            started_at=datetime.now()
        )
    if "frame_reader" not in st.session_state:
        st.session_state.frame_reader = None
    if "detector" not in st.session_state:
        st.session_state.detector = RippleDetector()
    if "health_monitor" not in st.session_state:
        st.session_state.health_monitor = StreamHealthMonitor()
    if "last_detection_time" not in st.session_state:
        st.session_state.last_detection_time = None
    if "last_detected_sequence" not in st.session_state:
        st.session_state.last_detected_sequence = None


def format_time_delta(start: datetime, end: datetime = None) -> str:
    """Format time difference in readable format."""
    if end is None:
        end = datetime.now()
    delta = end - start
    seconds = delta.total_seconds()

    if seconds < 1:
        return "< 1 sec"
    elif seconds < 60:
        return f"{int(seconds)} sec"
    else:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"


def render_processed_frame(frame, detection_result):
    """Render frame with detection visualization."""
    if frame is None:
        return None

    display_frame = frame.copy()

    if detection_result and hasattr(detection_result, 'visual_marker'):
        marker = detection_result.visual_marker
        if marker and "mark_regions" in marker:
            for region in marker["mark_regions"]:
                x, y, w, h = region["x"], region["y"], region["width"], region["height"]
                cv2.rectangle(
                    display_frame,
                    (x, y),
                    (x + w, y + h),
                    marker.get("color", (0, 255, 0)),
                    2
                )

        label_text = marker.get("label", "")
        cv2.putText(
            display_frame,
            label_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            marker.get("color", (0, 255, 0)),
            2
        )

    return display_frame


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Low-Latency Ripple Monitoring",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.title("🌊 Low-Latency Ripple Monitoring")

    if LOCAL_TRUSTED_NETWORK_ONLY:
        st.warning(
            "⚠️ This interface is designed for local, trusted networks only. "
            "Do not expose this application to public networks."
        )

    initialize_session_state()

    session = st.session_state.session
    detector = st.session_state.detector
    health_monitor = st.session_state.health_monitor

    col1, col2 = st.columns([3, 1])

    with col1:
        col_left, col_right = st.columns(2)
        with col_left:
            if st.button("▶️ Start", key="start_btn", use_container_width=True):
                if st.session_state.frame_reader is None:
                    st.session_state.frame_reader = LatestFrameReader(STREAM_URL)
                    st.session_state.frame_reader.start()
                session.started_at = datetime.now()
                session.stopped_at = None
                session.health_state = StreamHealthState.ACTIVE
                st.session_state.last_detection_time = None
                st.session_state.last_detected_sequence = None
                st.rerun()

        with col_right:
            if st.button("⏹️ Stop", key="stop_btn", use_container_width=True):
                if st.session_state.frame_reader:
                    st.session_state.frame_reader.stop()
                    st.session_state.frame_reader = None
                session.health_state = StreamHealthState.WAITING
                session.latest_frame = None
                session.latest_detection = None
                session.stopped_at = datetime.now()
                st.session_state.last_detection_time = None
                st.session_state.last_detected_sequence = None
                st.rerun()

    view_mode = st.radio(
        "View Mode",
        ["Processed", "Raw"],
        horizontal=True,
        key="view_mode"
    )
    session.view_mode = "processed" if view_mode == "Processed" else "raw"

    video_placeholder = st.empty()
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    detection_message_placeholder = st.empty()

    frame_reader = st.session_state.frame_reader

    if session.health_state == StreamHealthState.WAITING:
        st.info("Click **Start** to begin monitoring the stream.")

    elif session.health_state in MONITORING_STATES:
        if frame_reader is None:
            st.error("Frame reader not initialized. Click Start again.")

        else:
            while session.health_state in MONITORING_STATES:
                latest_frame = frame_reader.get_latest_frame()
                last_frame_time = frame_reader.get_last_frame_time()

                if latest_frame:
                    session.latest_frame = latest_frame

                    current_time = datetime.now()
                    time_since_frame = (
                        (current_time - last_frame_time).total_seconds()
                        if last_frame_time
                        else float('inf')
                    )

                    if time_since_frame > 3:
                        session.health_state = StreamHealthState.INTERRUPTED
                    else:
                        session.health_state = StreamHealthState.ACTIVE
                else:
                    if (datetime.now() - session.started_at).total_seconds() > 3:
                        session.health_state = StreamHealthState.INTERRUPTED

                detection_result = session.latest_detection
                if (
                    latest_frame
                    and latest_frame.sequence_number != st.session_state.last_detected_sequence
                ):
                    detection_result = detector.detect(latest_frame)
                    session.latest_detection = detection_result
                    st.session_state.last_detection_time = datetime.now()
                    st.session_state.last_detected_sequence = latest_frame.sequence_number

                    if detection_result.status.value == "unavailable":
                        session.health_state = StreamHealthState.DETECTION_UNAVAILABLE
                    elif session.health_state == StreamHealthState.DETECTION_UNAVAILABLE:
                        session.health_state = StreamHealthState.ACTIVE

                with status_placeholder.container():
                    st.subheader(f"Status: {session.health_state.value}")

                if latest_frame:
                    display_frame = latest_frame.image
                    if session.view_mode == "processed" and detection_result:
                        if detection_result.status.value == "available":
                            display_frame = render_processed_frame(
                                display_frame,
                                detection_result
                            )

                    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(display_frame_rgb, width="stretch")

                    with metrics_placeholder.container():
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                        with metric_col1:
                            frame_age = format_time_delta(latest_frame.captured_at)
                            st.metric("Frame Age", frame_age)

                        with metric_col2:
                            if detection_result and st.session_state.last_detection_time:
                                detection_age = format_time_delta(
                                    st.session_state.last_detection_time
                                )
                            else:
                                detection_age = "—"
                            st.metric("Detection Age", detection_age)

                        with metric_col3:
                            if detection_result and detection_result.status.value == "available":
                                st.metric(
                                    "Activity Score",
                                    f"{detection_result.activity_score}/100"
                                )
                            else:
                                st.metric("Activity Score", "—")

                        with metric_col4:
                            if detection_result and detection_result.status.value == "available":
                                level = detection_result.activity_level.value
                                st.metric("Activity Level", level)
                            else:
                                st.metric("Activity Level", "—")

                    if detection_result and detection_result.status.value == "unavailable":
                        detection_message_placeholder.warning(
                            f"Detection Unavailable: {detection_result.reason}"
                        )
                    else:
                        detection_message_placeholder.empty()

                time.sleep(0.05)

    st.divider()
    st.caption(
        "Low-latency ripple monitoring for aquaculture. "
        "For documentation and setup instructions, see specs/001-rtsp-ripple-monitoring/quickstart.md"
    )


if __name__ == "__main__":
    main()
