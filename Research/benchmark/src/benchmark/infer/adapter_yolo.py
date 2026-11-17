"""YOLO26 -> common prediction format (fixed conf floor, NMS IoU, max_det)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Config
from .common import Predictions, resolve_weights


def predict(cfg: Config, rec: dict[str, Any], images: list[tuple[int, Path]]) -> Predictions:
    from ultralytics import YOLO

    t = cfg.thresholds
    model = YOLO(str(resolve_weights(rec)))
    names = cfg.classes
    out: Predictions = []
    for image_id, path in images:
        r = model.predict(str(path), conf=t["ap_conf_floor"], iou=t["nms_iou"],
                          max_det=int(t["max_det"]), imgsz=int(rec["input_size"]),
                          verbose=False, device=0)[0]
        xyxy, conf, cls = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
        for (x0, y0, x1, y1), s, c in zip(xyxy, conf, cls):
            out.append({"image_id": image_id, "category": names[int(c)],
                        "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                        "score": float(s)})
    return out
