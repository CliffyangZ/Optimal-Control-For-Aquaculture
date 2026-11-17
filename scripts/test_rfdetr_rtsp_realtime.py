"""Realtime RF-DETR inference smoke test for the fish RTSP stream.

Default stream: rtsp://192.168.137.157:8554/cam
Default model: models/rf-detr-fish-regular.pth

Run from the project root:
    python scripts/test_rfdetr_rtsp_realtime.py

Press q in the preview window to stop.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "Depolyment" / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.infer.rfdetr_detector import RFDETRDetector


DEFAULT_RTSP_URL = "rtsp://127.0.0.1:8554/cam"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "rf-detr-fish-regular.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test realtime RF-DETR inference on an RTSP stream.")
    parser.add_argument("--rtsp-url", default=DEFAULT_RTSP_URL, help="RTSP stream URL")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="RF-DETR checkpoint path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device: cuda or cpu")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--width", type=int, default=0, help="Optional capture width; 0 keeps stream default")
    parser.add_argument("--height", type=int, default=0, help="Optional capture height; 0 keeps stream default")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 runs until q/Ctrl+C")
    parser.add_argument("--no-display", action="store_true", help="Run without cv2.imshow preview")
    return parser.parse_args()


def open_capture(rtsp_url: str, width: int, height: int) -> cv2.VideoCapture:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open RTSP stream: {rtsp_url}")
    return cap


def draw_detections(frame, detections) -> None:
    height, width = frame.shape[:2]
    for detection in detections:
        x1 = int(detection.bbox_x * width)
        y1 = int(detection.bbox_y * height)
        x2 = int((detection.bbox_x + detection.bbox_w) * width)
        y2 = int((detection.bbox_y + detection.bbox_h) * height)

        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def close_preview_window() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading RF-DETR model: {model_path}")
    print(f"Device: {args.device}, confidence threshold: {args.conf}")
    detector = RFDETRDetector.load(model_path, device=args.device, conf_threshold=args.conf)

    print(f"Opening RTSP stream: {args.rtsp_url}")
    cap = open_capture(args.rtsp_url, args.width, args.height)

    frame_id = 0
    fps_window: list[float] = []
    display_enabled = not args.no_display

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame; retrying...")
                time.sleep(0.2)
                continue

            frame_id += 1
            loop_start = time.monotonic()
            result = detector.detect(frame, frame_id=frame_id)
            elapsed = time.monotonic() - loop_start
            fps_window.append(elapsed)
            if len(fps_window) > 30:
                fps_window.pop(0)

            avg_fps = 1.0 / (sum(fps_window) / len(fps_window)) if fps_window else 0.0
            print(
                f"frame={frame_id} detections={len(result.detections)} "
                f"inference_ms={result.inference_ms:.1f} fps={avg_fps:.1f}",
                end="\r",
                flush=True,
            )

            if display_enabled:
                preview = frame.copy()
                draw_detections(preview, result.detections)
                cv2.putText(
                    preview,
                    f"FPS {avg_fps:.1f} | Infer {result.inference_ms:.1f} ms | Det {len(result.detections)}",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                try:
                    cv2.imshow("RF-DETR RTSP Realtime Test", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as exc:
                    display_enabled = False
                    print(
                        "\nOpenCV GUI preview is unavailable; continuing in --no-display mode. "
                        f"Original error: {exc}",
                        flush=True,
                    )

            if args.max_frames > 0 and frame_id >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        close_preview_window()

    print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
