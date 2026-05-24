"""RFDETRDetector: load an RF-DETR model and run per-frame inference.

Bounding boxes are returned normalized to [0.0, 1.0] relative to the source
frame dimensions (xyxy pixel coords → xywh normalized).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from config.setting import YOLO_CONFIDENCE_THRESHOLD
from core.types import Detection, DetectionResult


class RFDETRDetector:
    """Wraps a loaded RF-DETR model and exposes `detect(frame) -> DetectionResult`."""

    def __init__(self, model, conf_threshold: float) -> None:
        self._model = model
        self._conf_threshold = conf_threshold

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        device: str = "cpu",
        conf_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    ) -> "RFDETRDetector":
        """Load an RF-DETR model from checkpoint."""
        from rfdetr.detr import RFDETR

        model = RFDETR.from_checkpoint(str(model_path))
        model.device = torch.device(device)
        return cls(model, conf_threshold)

    def detect(self, raw_frame: np.ndarray, frame_id: int = 0) -> DetectionResult:
        """Run inference on a single BGR frame and return normalized detections."""
        t0 = time.monotonic()

        # RF-DETR expects RGB, convert from BGR
        rgb_frame = raw_frame[..., ::-1].copy()

        # Run prediction
        detections = self._model.predict(
            rgb_frame,
            threshold=self._conf_threshold
        )

        inference_ms = (time.monotonic() - t0) * 1000.0

        h, w = raw_frame.shape[:2]
        normalized_detections: list[Detection] = []

        # detections is a supervision.Detections object
        if detections is not None and len(detections) > 0:
            # Supervision returns pixel coordinates [x1, y1, x2, y2]
            boxes = detections.xyxy
            confidences = detections.confidence
            class_ids = detections.class_id

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if conf is None or conf < self._conf_threshold:
                    continue

                x1, y1, x2, y2 = box

                # Normalize pixel coordinates to [0, 1]
                bx = float(x1) / w
                by = float(y1) / h
                bw = float(x2 - x1) / w
                bh = float(y2 - y1) / h

                # Get class name from model if available
                class_name = str(int(cls_id)) if cls_id is not None else "unknown"
                if hasattr(self._model, 'class_names') and self._model.class_names:
                    if int(cls_id) < len(self._model.class_names):
                        class_name = self._model.class_names[int(cls_id)]

                normalized_detections.append(Detection(
                    class_name=class_name,
                    class_index=int(cls_id) if cls_id is not None else -1,
                    confidence=float(conf),
                    bbox_x=bx,
                    bbox_y=by,
                    bbox_w=bw,
                    bbox_h=bh,
                ))

        return DetectionResult(
            frame_id=frame_id,
            captured_at=datetime.now(timezone.utc),
            source_width=w,
            source_height=h,
            detections=normalized_detections,
            inference_ms=inference_ms,
        )
