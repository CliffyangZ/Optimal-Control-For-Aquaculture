"""RF-DETR -> common prediction format (top-300 by score, no NMS)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Config
from ..train.rfdetr import SIZE_CLASS
from .common import Predictions, resolve_weights


def class_names_from_ids(class_ids, classes: list[str]) -> list[str]:
    """rfdetr reports 0-based class indices (verified empirically on R-s: offset 0 gives AP50 0.55,
    offset -1 gives 0.02) even though the COCO export used ids 1..N."""
    return [classes[int(i)] for i in class_ids]


def predict(cfg: Config, rec: dict[str, Any], images: list[tuple[int, Path]]) -> Predictions:
    import numpy as np
    import rfdetr
    from PIL import Image

    t = cfg.thresholds
    model = getattr(rfdetr, SIZE_CLASS[rec["size"]])(
        pretrain_weights=str(resolve_weights(rec)), num_classes=len(cfg.classes))
    out: Predictions = []
    for image_id, path in images:
        det = model.predict(Image.open(path).convert("RGB"), threshold=float(t["ap_conf_floor"]),
                            include_source_image=False)
        if len(det) == 0:
            continue
        # the head has one spare output (index == n_classes); drop it before the top-k cut
        keep = np.where(det.class_id < len(cfg.classes))[0]
        order = keep[np.argsort(-det.confidence[keep], kind="stable")][: int(t["max_det"])]
        names = class_names_from_ids(det.class_id[order], cfg.classes)
        for k, name in zip(order, names):
            x0, y0, x1, y1 = (float(v) for v in det.xyxy[k])
            out.append({"image_id": image_id, "category": name,
                        "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(det.confidence[k])})
    return out
