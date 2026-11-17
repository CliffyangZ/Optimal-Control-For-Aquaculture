"""COCO AP / AR via pycocotools from the common prediction format."""
from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np

from ..data.export import CAT_OFFSET

_KEYS = ("ap_50_95", "ap_50", "ap_small", "ap_medium", "ap_large", "ar_300")
MAX_DETS = 300  # == thresholds.max_det


def _mean(arr) -> float | None:
    arr = arr[arr > -1]
    return None if arr.size == 0 else float(arr.mean())


def _stats(ev) -> dict[str, float | None]:
    """AP/AR from the accumulated arrays at maxDets = MAX_DETS.

    pycocotools' summarize() hard-codes maxDets=100 for AP, so it cannot be used here.
    precision: [T,R,K,A,M]; recall: [T,K,A,M]; A order: all, small, medium, large.
    """
    prec, rec = ev.eval["precision"], ev.eval["recall"]
    m = len(ev.params.maxDets) - 1
    t50 = int(np.where(np.isclose(ev.params.iouThrs, 0.5))[0][0])
    return {
        "ap_50_95": _mean(prec[:, :, :, 0, m]),
        "ap_50": _mean(prec[t50, :, :, 0, m]),
        "ap_small": _mean(prec[:, :, :, 1, m]),
        "ap_medium": _mean(prec[:, :, :, 2, m]),
        "ap_large": _mean(prec[:, :, :, 3, m]),
        "ar_300": _mean(rec[:, :, 0, m]),
    }


def to_coco_results(predictions: list[dict[str, Any]], classes: list[str]) -> list[dict[str, Any]]:
    cid = {n: i + CAT_OFFSET for i, n in enumerate(classes)}
    return [{"image_id": p["image_id"], "category_id": cid[p["category"]],
             "bbox": [float(v) for v in p["bbox"]], "score": float(p["score"])}
            for p in predictions]


def coco_ap(gt: dict[str, Any], predictions: list[dict[str, Any]], classes: list[str]
            ) -> dict[str, dict[str, float | None]]:
    """Return {"overall": {...}, "<class>": {...}} with ap_50_95, ap_50, ap_small/medium/large, ar_100.

    `gt` is COCO-style (data.export.gt_coco); `predictions` use the common format
    (image_id, category, bbox xywh, score). Classes without GT get None values.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cat_ids = [i + CAT_OFFSET for i in range(len(classes))]
    empty = {k: None for k in _KEYS}
    results = to_coco_results(predictions, classes)
    present = {a["category_id"] for a in gt["annotations"]}
    if not present:
        return {"overall": dict(empty), **{c: dict(empty) for c in classes}}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        if results:
            coco_dt = coco_gt.loadRes(results)
        else:
            coco_dt = None

    def run(cats: list[int]) -> dict[str, float | None]:
        if coco_dt is None or not (set(cats) & present):
            return dict(empty)
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(coco_gt, coco_dt, "bbox")
            ev.params.catIds = cats
            # images hold up to ~370 feed objects: the COCO default cap of 100 detections/image would bound recall
            ev.params.maxDets = [1, 10, MAX_DETS]
            ev.evaluate(); ev.accumulate(); ev.summarize()
        return _stats(ev)

    if coco_dt is None:  # no predictions at all: AP is zero where GT exists
        zero = {k: 0.0 for k in _KEYS}
        return {"overall": dict(zero),
                **{c: (dict(zero) if (i + CAT_OFFSET) in present else dict(empty))
                   for i, c in enumerate(classes)}}
    out = {"overall": run([c for c in cat_ids if c in present])}
    for i, name in enumerate(classes):
        out[name] = run([cat_ids[i]])
    return out
