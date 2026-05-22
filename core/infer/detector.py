"""YOLODetector: load a YOLO26 model and run per-frame inference.

Bounding boxes are returned normalized to [0.0, 1.0] relative to the source
frame dimensions (xyxy pixel coords → xywh normalized).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config.setting import YOLO_CONFIDENCE_THRESHOLD
from core.types import Detection, DetectionResult


class YOLODetector:
    """Wraps a loaded YOLO26 model and exposes `detect(frame) -> DetectionResult`."""

    def __init__(self, model, conf_threshold: float) -> None:
        self._model = model
        self._conf_threshold = conf_threshold

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        device: str = "cuda",
        conf_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    ) -> "YOLODetector":
        """Load a YOLO model from `model_path` onto `device`."""
        from ultralytics import YOLO  # ml group only

        model = YOLO(str(model_path))
        model.to(device)
        return cls(model, conf_threshold)

    def detect(self, raw_frame: np.ndarray) -> DetectionResult:
        """Run inference on a single BGR frame and return normalized detections."""
        t0 = time.monotonic()
        results = self._model(raw_frame, conf=self._conf_threshold, verbose=False)
        inference_ms = (time.monotonic() - t0) * 1000.0

        h, w = raw_frame.shape[:2]
        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                conf = float(box.conf[0].cpu())
                cls_idx = int(box.cls[0].cpu())
                cls_name = result.names.get(cls_idx, str(cls_idx))
                if conf < self._conf_threshold:
                    continue

                # Clip pixel coordinates before normalization so boxes crossing
                # frame edges keep the correct visible width and height.
                x1 = max(0.0, min(float(w), x1))
                y1 = max(0.0, min(float(h), y1))
                x2 = max(0.0, min(float(w), x2))
                y2 = max(0.0, min(float(h), y2))

                bx = x1 / w
                by = y1 / h
                bw = max(0.0, (x2 - x1) / w)
                bh = max(0.0, (y2 - y1) / h)

                detections.append(Detection(
                    class_name=cls_name,
                    class_index=cls_idx,
                    confidence=conf,
                    bbox_x=bx,
                    bbox_y=by,
                    bbox_w=bw,
                    bbox_h=bh,
                ))

        return DetectionResult(
            frame_id=0,
            captured_at=datetime.now(timezone.utc),
            source_width=w,
            source_height=h,
            detections=detections,
            inference_ms=inference_ms,
        )
