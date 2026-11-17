"""Evaluate one run on one eval set: predictions + GT -> metrics.json (detection block)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import Config
from ..infer.common import eval_set_gt
from ..infer.predictions import prediction_path, read_predictions
from ..records import read_record, set_status, write_record
from .coco_ap import coco_ap
from .prf_ci import prf_with_ci


def metrics_path(run_dir: Path, eval_set: str) -> Path:
    return run_dir / ("metrics.json" if eval_set == "heldout_real" else f"metrics.{eval_set}.json")


def evaluate_run(cfg: Config, run_id: str, eval_set: str = "heldout_real") -> dict[str, Any]:
    run_dir = cfg.run_dir(run_id)
    rec = read_record(run_dir)
    pred_file = prediction_path(run_dir, eval_set)
    if not pred_file.exists():
        raise FileNotFoundError(f"{pred_file} missing; run `predict --run {run_id} --eval-set {eval_set}`")
    doc = read_predictions(pred_file)
    gt = eval_set_gt(cfg, eval_set)
    t, bs = cfg.thresholds, cfg["bootstrap"]
    kpi = list(cfg["kpi_classes"])

    ap = coco_ap(gt, doc["predictions"], cfg.classes)
    prf = prf_with_ci(gt, doc["predictions"], cfg.classes, conf=float(t["conf"]),
                      iou_thr=float(t["eval_iou"]), B=int(bs["B"]), seed=int(bs["seed"]),
                      report_classes=kpi)
    detection = {g: {**ap[g], **prf[g]} for g in ["overall", *kpi]}
    metrics = {
        "run_id": run_id, "eval_set": eval_set, "n_images": len(gt["images"]),
        "n_gt": len(gt["annotations"]),
        "thresholds": {"conf": t["conf"], "eval_iou": t["eval_iou"],
                       "ap_conf_floor": t["ap_conf_floor"], "max_det": t["max_det"]},
        "bootstrap": {"B": bs["B"], "seed": bs["seed"], "unit": "image"},
        "detection": detection,
    }
    p = metrics_path(run_dir, eval_set)
    if p.exists():  # keep blocks written by other stages (e.g. `system` from `profile`)
        prev = json.loads(p.read_text(encoding="utf-8"))
        for k, v in prev.items():
            metrics.setdefault(k, v)
    p.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if eval_set == "heldout_real" and rec["status"] == "trained":
        write_record(run_dir, set_status(rec, "evaluated"))
    return metrics
