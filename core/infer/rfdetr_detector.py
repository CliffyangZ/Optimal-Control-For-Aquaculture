"""RFDETRDetector: load an RF-DETR model and run per-frame inference.

Bounding boxes are returned normalized to [0.0, 1.0] relative to the source
frame dimensions (xyxy pixel coords → xywh normalized).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from inspect import Parameter, signature
from pathlib import Path

import numpy as np
import torch

from config.setting import YOLO_CONFIDENCE_THRESHOLD
from core.types import Detection, DetectionResult


class RFDETRDetector:
    """Wraps a loaded RF-DETR model and exposes `detect(frame) -> DetectionResult`."""

    def __init__(
        self,
        model,
        conf_threshold: float,
        *,
        predict_shape: tuple[int, int] | None = None,
        include_source_image: bool = False,
    ) -> None:
        self._model = model
        self._conf_threshold = conf_threshold
        self._predict_shape = predict_shape
        self._include_source_image = include_source_image
        self._predict_accepts_kwargs = self._detect_predict_kwargs_support(model)

    @staticmethod
    def _detect_predict_kwargs_support(model) -> set[str] | None:
        try:
            parameters = signature(model.predict).parameters
        except (TypeError, ValueError):
            return None
        if any(param.kind == Parameter.VAR_KEYWORD for param in parameters.values()):
            return None
        return set(parameters)

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        device: str = "cpu",
        conf_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
        *,
        optimize: bool = True,
        compile: bool = True,
        dtype: str | torch.dtype = torch.float32,
        resolution: int | None = None,
        predict_shape: tuple[int, int] | None = None,
        include_source_image: bool = False,
    ) -> "RFDETRDetector":
        """Load an RF-DETR model from checkpoint."""
        import warnings
        import sys
        import os
        from io import StringIO

        from rfdetr.detr import RFDETR

        # Suppress all deprecation warnings and transformers logging
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', message='.*use_return_dict.*')

        # Suppress transformers debug output
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

        # Redirect stderr to suppress the printed warning
        old_stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            model = RFDETR.from_checkpoint(str(model_path))
        finally:
            sys.stderr = old_stderr

        # Set device in the model's internal ModelContext
        if hasattr(model, 'model') and hasattr(model.model, 'device'):
            model.model.device = torch.device(device)
        if resolution is not None and hasattr(model, 'model') and hasattr(model.model, 'resolution'):
            model.model.resolution = resolution

        if optimize and hasattr(model, 'optimize_for_inference'):
            model.optimize_for_inference(compile=compile, dtype=dtype)

        return cls(
            model,
            conf_threshold,
            predict_shape=predict_shape,
            include_source_image=include_source_image,
        )

    def detect(self, raw_frame: np.ndarray, frame_id: int = 0) -> DetectionResult:
        """Run inference on a single BGR frame and return normalized detections."""
        t0 = time.monotonic()

        # RF-DETR expects RGB, convert from BGR
        rgb_frame = raw_frame[..., ::-1].copy()

        predict_kwargs = {
            "threshold": self._conf_threshold,
            "include_source_image": self._include_source_image,
        }
        if self._predict_shape is not None:
            predict_kwargs["shape"] = self._predict_shape
        if self._predict_accepts_kwargs is not None:
            predict_kwargs = {
                key: value
                for key, value in predict_kwargs.items()
                if key in self._predict_accepts_kwargs
            }

        with torch.inference_mode():
            detections = self._model.predict(rgb_frame, **predict_kwargs)

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

                # Clip pixel coordinates before normalization so boxes crossing
                # frame edges keep the correct visible width and height.
                x1 = max(0.0, min(float(w), float(x1)))
                y1 = max(0.0, min(float(h), float(y1)))
                x2 = max(0.0, min(float(w), float(x2)))
                y2 = max(0.0, min(float(h), float(y2)))

                bx = x1 / w
                by = y1 / h
                bw = max(0.0, (x2 - x1) / w)
                bh = max(0.0, (y2 - y1) / h)

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
