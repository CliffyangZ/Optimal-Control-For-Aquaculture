"""D-FINE (Hugging Face) -> common prediction format (top-300 queries x classes, no NMS)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Config
from ..data.export import image_size
from ..train.dfine import decode_predictions
from .common import Predictions, resolve_weights


def predict(cfg: Config, rec: dict[str, Any], images: list[tuple[int, Path]],
            batch_size: int = 8) -> Predictions:
    import cv2
    import numpy as np
    import torch
    from transformers import DFineForObjectDetection

    t = cfg.thresholds
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DFineForObjectDetection.from_pretrained(str(resolve_weights(rec))).to(dev).eval()
    imgsz = int(rec["input_size"])
    out: Predictions = []
    for s in range(0, len(images), batch_size):
        chunk = images[s:s + batch_size]
        tensors, sizes = [], []
        for _, path in chunk:
            w, h = image_size(path)
            sizes.append((w, h))
            img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            tensors.append(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            res = model(pixel_values=torch.stack(tensors).to(dev))
        dec = decode_predictions(res.logits, res.pred_boxes, sizes, topk=int(t["max_det"]))
        for (image_id, _), rows in zip(chunk, dec):
            for c, sc, x, y, w, h in rows:
                if sc < t["ap_conf_floor"]:
                    continue
                out.append({"image_id": image_id, "category": cfg.classes[int(c)],
                            "bbox": [float(x), float(y), float(w), float(h)], "score": float(sc)})
    return out
