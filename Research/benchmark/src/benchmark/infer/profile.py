"""Latency / VRAM profiler (batch=1, fp32, same image for every model).

model_only : forward pass on an already preprocessed GPU tensor (cuda-synchronised).
end_to_end : decoded image in memory -> preprocessing -> forward -> post-processing -> detections.
Disk reads are excluded from both. Wall-clock fields are expected to differ between runs.
"""
from __future__ import annotations

import gc
import platform
import subprocess
from typing import Any, Callable

import numpy as np

from ..config import REPO_ROOT, Config
from .common import eval_set_images, resolve_weights


def gpu_state() -> dict[str, Any]:
    try:
        q = "name,pstate,power.draw,clocks.sm,clocks.max.sm,temperature.gpu"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return dict(zip(q.split(","), [v.strip() for v in out.split(",")]))
    except Exception:  # noqa: BLE001
        return {}


def time_ms(fn: Callable[[], Any], warmup: int, iters: int) -> list[float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = __import__("time").perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((__import__("time").perf_counter() - t0) * 1000.0)
    return ts


def summarize(ts: list[float]) -> dict[str, float]:
    a = np.asarray(ts)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "mean": float(a.mean())}


def _yolo_fns(cfg: Config, rec: dict[str, Any], img_bgr: np.ndarray):
    import torch
    from ultralytics import YOLO

    t, size = cfg.thresholds, int(rec["input_size"])
    model = YOLO(str(resolve_weights(rec)))
    net = model.model.to("cuda").eval().float().fuse(verbose=False)  # Ultralytics fuses Conv+BN at predict time
    x = torch.rand(1, 3, size, size, device="cuda")

    def model_only():
        with torch.no_grad():
            net(x)

    def e2e():
        model.predict(img_bgr, conf=t["ap_conf_floor"], iou=t["nms_iou"], max_det=int(t["max_det"]),
                      imgsz=size, verbose=False, device=0)
    return model_only, e2e, model


def _rfdetr_fns(cfg: Config, rec: dict[str, Any], img_bgr: np.ndarray):
    import rfdetr
    import torch
    from PIL import Image

    from ..train.rfdetr import SIZE_CLASS

    t = cfg.thresholds
    model = getattr(rfdetr, SIZE_CLASS[rec["size"]])(
        pretrain_weights=str(resolve_weights(rec)), num_classes=len(cfg.classes))
    net = model.model.model.to("cuda").eval().float()
    res = int(model.model.resolution)
    x = torch.rand(1, 3, res, res, device="cuda")
    pil = Image.fromarray(img_bgr[:, :, ::-1].copy())

    def model_only():
        with torch.no_grad():
            net(x)

    def e2e():
        model.predict(pil, threshold=float(t["ap_conf_floor"]), include_source_image=False)
    return model_only, e2e, model


def profile_run(cfg: Config, rec: dict[str, Any]) -> dict[str, Any]:
    import cv2
    import torch

    fam = rec["family"]
    if fam == "dfine":
        raise NotImplementedError("D-FINE is excluded from profiling for now")
    if not torch.cuda.is_available():
        raise RuntimeError("profiling needs a CUDA GPU")
    _, path = eval_set_images(cfg, "heldout_real")[0]
    img = cv2.imread(str(path))
    p = cfg["profile"]
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    build = _yolo_fns if fam == "yolo26" else _rfdetr_fns
    model_only, e2e, keep = build(cfg, rec, img)
    state_before = gpu_state()
    mo = time_ms(model_only, int(p["warmup"]), int(p["iters"]))
    ee = time_ms(e2e, int(p["warmup"]), int(p["iters"]))
    state_after = gpu_state()
    peak = torch.cuda.max_memory_allocated() / 2**20
    del keep
    gc.collect(); torch.cuda.empty_cache()
    mo_s, ee_s = summarize(mo), summarize(ee)
    return {
        "latency_ms_p50": mo_s["p50"], "latency_ms_p95": mo_s["p95"],
        "latency_ms_e2e_p50": ee_s["p50"], "latency_ms_e2e_p95": ee_s["p95"],
        "fps_effective": 1000.0 / ee_s["p50"],
        "vram_peak_mb": float(peak),
        "device": torch.cuda.get_device_name(0),
        "gpu_power_state": {"before": state_before, "after": state_after},
        "protocol": {"batch": 1, "precision": "fp32", "warmup": int(p["warmup"]),
                     "iters": int(p["iters"]), "image": str(path.relative_to(REPO_ROOT)),
                     "input_size": int(rec["input_size"]),
                     "note": "RF-DETR not optimize_for_inference; YOLO plain PyTorch"},
        "host": platform.node(),
    }
