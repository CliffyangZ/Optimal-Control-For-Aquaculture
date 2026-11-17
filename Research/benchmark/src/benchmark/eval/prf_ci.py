"""Precision / recall / F1 at a fixed confidence with image-level bootstrap CIs."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..data.export import CAT_OFFSET


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xywh boxes -> (N,M)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_image(gt_boxes: np.ndarray, pred_boxes: np.ndarray, pred_scores: np.ndarray,
                iou_thr: float) -> tuple[int, int, int]:
    """Greedy one-to-one matching by descending score; returns (tp, fp, fn)."""
    ng, npd = len(gt_boxes), len(pred_boxes)
    if npd == 0:
        return 0, 0, ng
    if ng == 0:
        return 0, npd, 0
    order = np.argsort(-np.asarray(pred_scores), kind="stable")
    iou = iou_xywh(pred_boxes, gt_boxes)
    used = np.zeros(ng, dtype=bool)
    tp = 0
    for p in order:
        cand = np.where(~used & (iou[p] >= iou_thr))[0]
        if len(cand):
            j = cand[np.argmax(iou[p, cand])]
            used[j] = True
            tp += 1
    return tp, npd - tp, ng - tp


def per_image_counts(gt: dict[str, Any], predictions: list[dict[str, Any]], classes: list[str],
                     conf: float, iou_thr: float) -> tuple[np.ndarray, list[Any]]:
    """Return counts (N_images, n_classes, 3=[tp,fp,fn]) and the image-id order. Class-aware matching."""
    ids = [im["id"] for im in gt["images"]]
    row = {i: k for k, i in enumerate(ids)}
    cid = {n: i + CAT_OFFSET for i, n in enumerate(classes)}
    g: dict[tuple[int, int], list] = {}
    for a in gt["annotations"]:
        g.setdefault((row[a["image_id"]], a["category_id"]), []).append(a["bbox"])
    p: dict[tuple[int, int], list] = {}
    for d in predictions:
        if d["score"] < conf:
            continue
        p.setdefault((row[d["image_id"]], cid[d["category"]]), []).append((d["bbox"], d["score"]))
    counts = np.zeros((len(ids), len(classes), 3), dtype=np.int64)
    for key in set(g) | set(p):
        r, c = key
        gb = np.array(g.get(key, []), dtype=float).reshape(-1, 4)
        pl = p.get(key, [])
        pb = np.array([x[0] for x in pl], dtype=float).reshape(-1, 4)
        ps = np.array([x[1] for x in pl], dtype=float)
        counts[r, c - CAT_OFFSET] = match_image(gb, pb, ps, iou_thr)
    return counts, ids


def prf_from_totals(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray):
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1


def _group_totals(counts: np.ndarray, groups: dict[str, list[int]]) -> np.ndarray:
    """(..., n_classes, 3) -> (..., n_groups, 3) summing class columns per group."""
    return np.stack([counts[..., idx, :].sum(axis=-2) for idx in groups.values()], axis=-2)


def prf_with_ci(gt: dict[str, Any], predictions: list[dict[str, Any]], classes: list[str],
                conf: float, iou_thr: float, B: int, seed: int,
                report_classes: list[str] | None = None, chunk: int = 250
                ) -> dict[str, dict[str, dict[str, float]]]:
    """{"overall"|class: {"precision"|"recall"|"f1": {"value","ci_low","ci_high"}}}.

    "overall" is micro-averaged over all classes. Bootstrap resamples images (the unit),
    seeded, percentile 95% interval.
    """
    counts, _ = per_image_counts(gt, predictions, classes, conf, iou_thr)
    n_img = counts.shape[0]
    groups = {"overall": list(range(len(classes)))}
    for c in (report_classes or classes):
        groups[c] = [classes.index(c)]
    point = _group_totals(counts.sum(axis=0), groups)  # (G,3)
    pm = prf_from_totals(point[:, 0], point[:, 1], point[:, 2])
    boots = [[], [], []]
    rng = np.random.default_rng(seed)
    if n_img:
        done = 0
        while done < B:
            m = min(chunk, B - done)
            idx = rng.integers(0, n_img, size=(m, n_img))
            tot = _group_totals(counts[idx].sum(axis=1), groups)  # (m,G,3)
            r = prf_from_totals(tot[..., 0], tot[..., 1], tot[..., 2])
            for k in range(3):
                boots[k].append(r[k])
            done += m
    out: dict[str, dict[str, dict[str, float]]] = {}
    for gi, name in enumerate(groups):
        out[name] = {}
        for k, metric in enumerate(("precision", "recall", "f1")):
            val = float(pm[k][gi])
            if n_img:
                col = np.concatenate(boots[k], axis=0)[:, gi]
                lo, hi = np.percentile(col, [2.5, 97.5])
            else:
                lo = hi = val
            out[name][metric] = {"value": val, "ci_low": float(lo), "ci_high": float(hi)}
    return out
