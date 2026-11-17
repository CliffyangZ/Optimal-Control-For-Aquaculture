"""Benchmark RF-DETR inference latency and FPS.

Default model path: models/rf-detr-fish-regular.pth
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_SRC = Path(__file__).resolve().parent / "Depolyment" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.infer.rfdetr_detector import RFDETRDetector


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "rf-detr-fish-regular.pth"
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "fish_feeding.mp4"
TARGET_LATENCY_MS = 33.3


@dataclass(frozen=True)
class BenchmarkStats:
    frames: int
    detections: int
    total_ms_mean: float
    total_ms_median: float
    total_ms_p90: float
    total_ms_p95: float
    total_ms_min: float
    total_ms_max: float
    inference_ms_mean: float
    inference_ms_median: float
    read_ms_mean: float
    resize_ms_mean: float

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms_mean if self.total_ms_mean > 0 else 0.0

    @property
    def meets_target(self) -> bool:
        return self.total_ms_mean < TARGET_LATENCY_MS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RF-DETR inference latency/FPS benchmark")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="RF-DETR .pth checkpoint path")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE), help="Video, image, RTSP URL, camera index, or 'synthetic'")
    parser.add_argument("--device", default=default_device(), help="Torch device: auto, cuda, mps, cpu, or cuda index")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--max-frames", type=int, default=120, help="Measured frames after warmup")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup frames excluded from stats")
    parser.add_argument("--width", type=int, default=0, help="Resize frame width before detection; 0 keeps source size")
    parser.add_argument("--height", type=int, default=0, help="Resize frame height before detection; 0 keeps source size")
    parser.add_argument("--resolution", type=int, default=0, help="Override square RF-DETR model input resolution before optimize")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"), help="Optimized inference dtype")
    parser.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=True, help="Use RF-DETR optimize_for_inference")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False, help="JIT-trace optimized RF-DETR model")
    parser.add_argument("--synthetic-size", default="640x480", help="Synthetic source size as WIDTHxHEIGHT")
    parser.add_argument("--json", type=Path, default=None, help="Optional path for JSON benchmark report")
    return parser.parse_args(argv)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: str) -> str:
    if device == "auto":
        return default_device()
    if device.isdigit():
        return f"cuda:{device}"
    return device


def resolve_dtype(dtype: str, device: str) -> str:
    if dtype != "auto":
        return dtype
    return "float16" if device.startswith("cuda") else "float32"


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def open_source(source: str) -> cv2.VideoCapture | None:
    if source == "synthetic":
        return None
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return cap


def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT, for example 640x480") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return width, height


def synthetic_frame(frame_id: int, width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    center_x = (frame_id * 7) % width
    center_y = height // 2
    cv2.circle(frame, (center_x, center_y), max(8, min(width, height) // 16), (120, 190, 80), -1)
    cv2.rectangle(frame, (width // 4, height // 4), (width // 2, height // 2), (30, 80, 180), 2)
    return frame


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def compute_stats(samples: list[dict[str, float]]) -> BenchmarkStats:
    total_ms = [sample["total_ms"] for sample in samples]
    inference_ms = [sample["inference_ms"] for sample in samples]
    read_ms = [sample["read_ms"] for sample in samples]
    resize_ms = [sample["resize_ms"] for sample in samples]
    detections = int(sum(sample["detections"] for sample in samples))
    return BenchmarkStats(
        frames=len(samples),
        detections=detections,
        total_ms_mean=statistics.fmean(total_ms),
        total_ms_median=statistics.median(total_ms),
        total_ms_p90=percentile(total_ms, 90),
        total_ms_p95=percentile(total_ms, 95),
        total_ms_min=min(total_ms),
        total_ms_max=max(total_ms),
        inference_ms_mean=statistics.fmean(inference_ms),
        inference_ms_median=statistics.median(inference_ms),
        read_ms_mean=statistics.fmean(read_ms),
        resize_ms_mean=statistics.fmean(resize_ms),
    )


def resize_frame(frame: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float]:
    if width <= 0 and height <= 0:
        return frame, 0.0
    source_h, source_w = frame.shape[:2]
    target_w = width if width > 0 else int(round(source_w * (height / source_h)))
    target_h = height if height > 0 else int(round(source_h * (width / source_w)))
    t0 = time.perf_counter()
    resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized, (time.perf_counter() - t0) * 1000.0


def read_frame(cap: cv2.VideoCapture | None, frame_id: int, synthetic_size: tuple[int, int]) -> tuple[bool, np.ndarray | None, float]:
    t0 = time.perf_counter()
    if cap is None:
        frame = synthetic_frame(frame_id, *synthetic_size)
        return True, frame, (time.perf_counter() - t0) * 1000.0
    ok, frame = cap.read()
    return ok, frame, (time.perf_counter() - t0) * 1000.0


def stats_to_dict(stats: BenchmarkStats) -> dict[str, Any]:
    return {
        "frames": stats.frames,
        "detections": stats.detections,
        "fps": round(stats.fps, 2),
        "target_latency_ms": TARGET_LATENCY_MS,
        "meets_target": stats.meets_target,
        "total_ms": {
            "mean": round(stats.total_ms_mean, 2),
            "median": round(stats.total_ms_median, 2),
            "p90": round(stats.total_ms_p90, 2),
            "p95": round(stats.total_ms_p95, 2),
            "min": round(stats.total_ms_min, 2),
            "max": round(stats.total_ms_max, 2),
        },
        "model_inference_ms": {
            "mean": round(stats.inference_ms_mean, 2),
            "median": round(stats.inference_ms_median, 2),
        },
        "overhead_ms": {
            "read_mean": round(stats.read_ms_mean, 2),
            "resize_mean": round(stats.resize_ms_mean, 2),
        },
    }


def print_stats(stats: BenchmarkStats) -> None:
    print("\nBenchmark result")
    print(f"frames={stats.frames} detections={stats.detections}")
    print(f"fps={stats.fps:.2f}")
    print(
        "end_to_end_ms "
        f"mean={stats.total_ms_mean:.2f} median={stats.total_ms_median:.2f} "
        f"p90={stats.total_ms_p90:.2f} p95={stats.total_ms_p95:.2f} "
        f"min={stats.total_ms_min:.2f} max={stats.total_ms_max:.2f}"
    )
    print(f"model_inference_ms mean={stats.inference_ms_mean:.2f} median={stats.inference_ms_median:.2f}")
    print(f"overhead_ms read_mean={stats.read_ms_mean:.2f} resize_mean={stats.resize_ms_mean:.2f}")
    status = "PASS" if stats.meets_target else "FAIL"
    print(f"target end-to-end < {TARGET_LATENCY_MS:.1f} ms: {status}")


def validate_model_path(model_path: Path) -> None:
    if model_path.exists():
        return
    typo_path = PROJECT_ROOT / "models" / "rf-detr-fish-regular.pth"
    if model_path == typo_path and DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Did you mean {DEFAULT_MODEL_PATH}?")
    raise FileNotFoundError(f"Model not found: {model_path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_path = args.model.resolve()
    validate_model_path(model_path)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    synthetic_size = parse_size(args.synthetic_size)
    resolution = args.resolution if args.resolution > 0 else None

    print(f"model={model_path}")
    print(f"source={args.source}")
    print(f"device={device} dtype={dtype} optimize={args.optimize} compile={args.compile} resolution={resolution}")
    if args.width > 0 or args.height > 0:
        print(f"frame_resize={args.width or 'auto'}x{args.height or 'auto'}")

    load_start = time.perf_counter()
    detector = RFDETRDetector.load(
        model_path,
        device=device,
        conf_threshold=args.conf,
        optimize=args.optimize,
        compile=args.compile,
        dtype=dtype,
        resolution=resolution,
        include_source_image=False,
    )
    load_ms = (time.perf_counter() - load_start) * 1000.0
    print(f"load_ms={load_ms:.1f}")

    cap = open_source(args.source)
    samples: list[dict[str, float]] = []
    total_frames = args.warmup + args.max_frames
    frame_id = 0

    try:
        while frame_id < total_frames:
            frame_start = time.perf_counter()
            ok, frame, read_ms = read_frame(cap, frame_id, synthetic_size)
            if not ok or frame is None:
                break
            frame_id += 1

            frame, resize_ms = resize_frame(frame, args.width, args.height)
            synchronize(device)
            result = detector.detect(frame, frame_id=frame_id)
            synchronize(device)
            total_ms = (time.perf_counter() - frame_start) * 1000.0

            if frame_id > args.warmup:
                samples.append(
                    {
                        "total_ms": total_ms,
                        "inference_ms": result.inference_ms,
                        "read_ms": read_ms,
                        "resize_ms": resize_ms,
                        "detections": float(len(result.detections)),
                    }
                )

            phase = "warmup" if frame_id <= args.warmup else "measure"
            measured = max(0, frame_id - args.warmup)
            print(
                f"{phase} frame={frame_id}/{total_frames} measured={measured}/{args.max_frames} "
                f"total_ms={total_ms:.1f} inference_ms={result.inference_ms:.1f} detections={len(result.detections)}",
                end="\r",
                flush=True,
            )
    finally:
        if cap is not None:
            cap.release()
    print()

    if not samples:
        raise RuntimeError("No measured frames were processed")

    stats = compute_stats(samples)
    print_stats(stats)
    report = {
        "model": str(model_path),
        "source": args.source,
        "device": device,
        "dtype": dtype,
        "confidence": args.conf,
        "optimize": args.optimize,
        "compile": args.compile,
        "resolution": resolution,
        "frame_resize": {"width": args.width, "height": args.height},
        "load_ms": round(load_ms, 2),
        "stats": stats_to_dict(stats),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        print(f"[inference] ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"json_report={args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
