"""Low-latency YOLO pose estimation Streamlit interface."""

import cv2
import streamlit as st
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from config.setting import (
    LOCAL_TRUSTED_NETWORK_ONLY,
    POSE_CONFIDENCE_THRESHOLD,
    POSE_KEYPOINT_CONFIDENCE_THRESHOLD,
    POSE_YOLO_MODEL_PATH,
    STREAM_URL,
    TARGET_DETECTION_FPS,
    TARGET_DISPLAY_FPS,
)
from core.models import MonitoringSession, StreamHealthState
from core.pose_estimation import YoloPoseEstimator
from streaming.latest_frame_reader import LatestFrameReader


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
    if "pose_estimator" not in st.session_state:
        st.session_state.pose_estimator = YoloPoseEstimator(
            model_path=POSE_YOLO_MODEL_PATH,
            confidence_threshold=POSE_CONFIDENCE_THRESHOLD,
            keypoint_confidence_threshold=POSE_KEYPOINT_CONFIDENCE_THRESHOLD,
        )
    if "last_recognition_time" not in st.session_state:
        st.session_state.last_recognition_time = None
    if "last_recognized_sequence" not in st.session_state:
        st.session_state.last_recognized_sequence = None
    if "last_recognition_attempt_time" not in st.session_state:
        st.session_state.last_recognition_attempt_time = 0.0
    if "last_displayed_sequence" not in st.session_state:
        st.session_state.last_displayed_sequence = None


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


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="YOLOv26 Pose Estimation",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.title("YOLOv26 Pose Estimation")

    if LOCAL_TRUSTED_NETWORK_ONLY:
        st.warning(
            "⚠️ This interface is designed for local, trusted networks only. "
            "Do not expose this application to public networks."
        )

    initialize_session_state()

    session = st.session_state.session
    pose_estimator = st.session_state.pose_estimator
    model_path = Path(POSE_YOLO_MODEL_PATH)
    model_missing = model_path.suffix == ".pt" and not model_path.exists()

    if model_missing:
        st.error(
            f"YOLOv26 model not found: `{POSE_YOLO_MODEL_PATH}`. "
            "Place the weights file there, or set `POSE_YOLO_MODEL_PATH` to your model path."
        )

    col1, col2 = st.columns([3, 1])

    with col1:
        col_left, col_right = st.columns(2)
        with col_left:
            if st.button(
                "▶️ Start",
                key="start_btn",
                use_container_width=True,
                disabled=model_missing,
            ):
                if st.session_state.frame_reader is None:
                    st.session_state.frame_reader = LatestFrameReader(STREAM_URL)
                    st.session_state.frame_reader.start()
                session.started_at = datetime.now()
                session.stopped_at = None
                session.health_state = StreamHealthState.ACTIVE
                st.session_state.last_recognition_time = None
                st.session_state.last_recognized_sequence = None
                st.session_state.last_recognition_attempt_time = 0.0
                st.session_state.last_displayed_sequence = None
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
                st.session_state.last_recognition_time = None
                st.session_state.last_recognized_sequence = None
                st.session_state.last_recognition_attempt_time = 0.0
                st.session_state.last_displayed_sequence = None
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
        st.info("Click **Start** to begin pose estimation on the stream.")

    elif session.health_state in MONITORING_STATES:
        if frame_reader is None:
            st.error("Frame reader not initialized. Click Start again.")

        else:
            display_interval_seconds = 1 / TARGET_DISPLAY_FPS if TARGET_DISPLAY_FPS > 0 else 0
            detection_interval_seconds = (
                1 / TARGET_DETECTION_FPS if TARGET_DETECTION_FPS > 0 else 0
            )

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

                pose_result = session.latest_detection
                now_monotonic = time.monotonic()
                if (
                    latest_frame
                    and latest_frame.sequence_number != st.session_state.last_recognized_sequence
                    and (
                        now_monotonic - st.session_state.last_recognition_attempt_time
                        >= detection_interval_seconds
                    )
                ):
                    st.session_state.last_recognition_attempt_time = now_monotonic
                    pose_result = pose_estimator.estimate(latest_frame)
                    session.latest_detection = pose_result
                    st.session_state.last_recognition_time = datetime.now()
                    st.session_state.last_recognized_sequence = latest_frame.sequence_number

                    if not pose_result.available:
                        session.health_state = StreamHealthState.DETECTION_UNAVAILABLE
                    elif session.health_state == StreamHealthState.DETECTION_UNAVAILABLE:
                        session.health_state = StreamHealthState.ACTIVE

                with status_placeholder.container():
                    st.subheader(f"Status: {session.health_state.value}")

                if (
                    latest_frame
                    and latest_frame.sequence_number != st.session_state.last_displayed_sequence
                ):
                    display_frame = latest_frame.image
                    if session.view_mode == "processed" and pose_result:
                        if pose_result.available:
                            display_frame = pose_estimator.draw(display_frame, pose_result)

                    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(display_frame_rgb, width="stretch")
                    st.session_state.last_displayed_sequence = latest_frame.sequence_number

                    with metrics_placeholder.container():
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                        with metric_col1:
                            frame_age = format_time_delta(latest_frame.captured_at)
                            st.metric("Frame Age", frame_age)

                        with metric_col2:
                            if pose_result and st.session_state.last_recognition_time:
                                recognition_age = format_time_delta(
                                    st.session_state.last_recognition_time
                                )
                            else:
                                recognition_age = "—"
                            st.metric("Pose Age", recognition_age)

                        with metric_col3:
                            if pose_result and pose_result.available:
                                st.metric("Skeletons", len(pose_result.skeletons))
                            else:
                                st.metric("Skeletons", "—")

                        with metric_col4:
                            if pose_result and pose_result.available:
                                keypoint_count = sum(
                                    len(skeleton.keypoints)
                                    for skeleton in pose_result.skeletons
                                )
                                st.metric("Keypoints", keypoint_count)
                            else:
                                st.metric("Keypoints", "—")

                    if pose_result and not pose_result.available:
                        detection_message_placeholder.warning(
                            f"Pose Estimation Unavailable: {pose_result.reason}"
                        )
                    elif pose_result and pose_result.available and pose_result.skeletons:
                        with detection_message_placeholder.container():
                            st.subheader("Skeleton Coordinates")
                            for skeleton in pose_result.skeletons:
                                st.caption(f"Person {skeleton.person_id}")
                                st.json(skeleton.coordinates(), expanded=False)
                    else:
                        detection_message_placeholder.empty()

                time.sleep(display_interval_seconds)

    st.divider()
    st.caption(
        "Local YOLOv26 pose estimation for trusted-network video streams. "
        f"Model: {POSE_YOLO_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
