"""CLI entry point for the YOLO Fish & Feed Detection pipeline.

Subcommands:
    download   Download FishTracker_v2 from Roboflow
    train      Train YOLO26 with wandb logging
    evaluate   Evaluate trained model and apply the quality gate
    infer      Run inference on an image, video, or RTSP stream
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.config import load_training_config


def _cmd_download(args: argparse.Namespace) -> int:
    from core.train.dataset import FishTrackerDataset

    config = load_training_config(args.config)
    dataset = FishTrackerDataset(config)

    try:
        cache_dir = dataset.download()
        summary = dataset.validate()
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(f"[download] ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"[download] Dataset cached: {cache_dir} "
        f"({summary.image_count} images, {len(summary.classes)} classes)"
    )
    print(f"[download] data.yaml: {summary.dataset_yaml}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from core.train.evaluator import Evaluator

    config = load_training_config(args.config)
    # Use --runs-dir if provided, otherwise derive from config.model.output_dir
    if args.runs_dir is not None:
        runs_dir = Path(args.runs_dir)
    else:
        runs_dir = Path(config.model.output_dir)

    # Find the most recent run (latest timestamp)
    run_dirs = sorted(runs_dir.glob("run_*"))
    if not run_dirs:
        print(
            f"[evaluate] ERROR: no training runs found in {runs_dir}. "
            f"Run `python train.py train` first.",
            file=sys.stderr,
        )
        return 1

    latest_run = run_dirs[-1]
    best_pt = latest_run / "weights" / "best.pt"
    if not best_pt.exists():
        print(
            f"[evaluate] ERROR: best.pt not found at {best_pt}",
            file=sys.stderr,
        )
        return 1

    # Find data.yaml for validation
    cache_dir = Path(config.dataset.local_cache_dir)
    data_yaml = None
    for candidate in [cache_dir / "data.yaml", cache_dir / "dataset.yaml"]:
        if candidate.exists():
            data_yaml = candidate
            break
    if not data_yaml:
        for p in cache_dir.rglob("data.yaml"):
            data_yaml = p
            break

    if not data_yaml or not data_yaml.exists():
        print(
            f"[evaluate] ERROR: data.yaml not found in {cache_dir}",
            file=sys.stderr,
        )
        return 1

    evaluator = Evaluator(config.evaluation)
    print(f"[evaluate] Evaluating — {best_pt}", flush=True)
    report = evaluator.evaluate(best_pt, data_yaml)

    report_path = runs_dir.parent / "evaluation_report.json"
    evaluator.write_report(report, report_path)
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    import json as _json
    import cv2

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[infer] ERROR: model file not found: {model_path}", file=sys.stderr)
        return 1

    # Determine model type: auto-detect or use explicit type
    model_type = args.model_type
    if model_type == "auto":
        # Auto-detect based on file path patterns
        model_name = str(model_path).lower()
        if "rfdetr" in model_name or "rf-detr" in model_name:
            model_type = "rfdetr"
        elif model_path.suffix == ".pt":
            model_type = "yolo"
        else:
            model_type = "yolo"  # Default to YOLO

    if model_type == "rfdetr":
        from core.infer.rfdetr_detector import RFDETRDetector
        detector = RFDETRDetector.load(
            model_path,
            device=args.device,
            conf_threshold=args.conf,
        )
    else:
        from core.infer.detector import YOLODetector
        detector = YOLODetector.load(
            model_path,
            device=args.device,
            conf_threshold=args.conf,
        )

    source = args.source
    cap = cv2.VideoCapture(source if source.startswith("rtsp") else str(source))
    if not cap.isOpened():
        print(f"[infer] ERROR: cannot open source: {source}", file=sys.stderr)
        return 1

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Setup video writer if output path provided
    writer = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"[infer] Writing output to: {output_path}", flush=True)

    # Class colors for visualization
    colors = {
        "fish": (0, 255, 0),      # Green
        "feed": (255, 0, 0),      # Blue
        "debris": (0, 165, 255),  # Orange
        "lumpfish": (128, 0, 128) # Purple
    }

    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            result = detector.detect(frame, frame_id=frame_id)

            # Draw detections on frame
            for det in result.detections:
                x1 = int(det.bbox_x * width)
                y1 = int(det.bbox_y * height)
                x2 = int((det.bbox_x + det.bbox_w) * width)
                y2 = int((det.bbox_y + det.bbox_h) * height)

                color = colors.get(det.class_name, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{det.class_name} {det.confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Output JSON detection
                print(
                    _json.dumps({
                        "frame_id": frame_id,
                        "class_name": det.class_name,
                        "confidence": round(det.confidence, 4),
                        "x": round(det.bbox_x, 4),
                        "y": round(det.bbox_y, 4),
                        "width": round(det.bbox_w, 4),
                        "height": round(det.bbox_h, 4),
                    }),
                    flush=True,
                )

            # Write frame to output video
            if writer:
                writer.write(frame)

            # For image files, stop after first frame
            if not source.startswith("rtsp") and not str(source).endswith((".mp4", ".avi", ".mov")):
                break

        if writer:
            print(f"[infer] Output video complete: {args.output}", flush=True)
    finally:
        cap.release()
        if writer:
            writer.release()
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from core.train.callbacks import setup_wandb_mode
    from core.train.trainer import Trainer

    config = load_training_config(args.config)
    setup_wandb_mode(config.wandb, disabled=args.no_wandb)

    # Find data.yaml in the local cache directory
    cache_dir = Path(config.dataset.local_cache_dir)
    if not cache_dir.exists():
        print(
            f"[train] ERROR: dataset cache not found: {cache_dir}. "
            f"Run `python train.py download` first.",
            file=sys.stderr,
        )
        return 1

    # Look for data.yaml in the cache directory (Roboflow's data structure)
    data_yaml = None
    for candidate in [cache_dir / "data.yaml", cache_dir / "dataset.yaml"]:
        if candidate.exists():
            data_yaml = candidate
            break
    if not data_yaml:
        # Search recursively as Roboflow sometimes nests it
        for p in cache_dir.rglob("data.yaml"):
            data_yaml = p
            break

    if not data_yaml or not data_yaml.exists():
        print(
            f"[train] ERROR: data.yaml not found in {cache_dir}. "
            f"Run `python train.py download` first.",
            file=sys.stderr,
        )
        return 1

    trainer = Trainer(config, project_root=Path.cwd())
    try:
        trainer.train(data_yaml)
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(f"[train] ERROR: {exc}", file=sys.stderr)
        return 2

    print("[train] Training complete. Run `python train.py evaluate` to evaluate the model.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train.py", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download dataset from Roboflow")
    p_dl.add_argument("--config", type=Path, default=Path("config/training.yaml"))
    p_dl.set_defaults(func=_cmd_download)

    p_tr = sub.add_parser("train", help="Train YOLO26 with wandb logging")
    p_tr.add_argument("--config", type=Path, default=Path("config/training.yaml"))
    p_tr.add_argument(
        "--no-wandb", action="store_true", help="Disable wandb logging entirely"
    )
    p_tr.set_defaults(func=_cmd_train)

    p_ev = sub.add_parser("evaluate", help="Evaluate model and apply quality gate")
    p_ev.add_argument("--config", type=Path, default=Path("config/training.yaml"))
    p_ev.add_argument("--runs-dir", type=Path, default=None, help="Override output directory (default: from config)")
    p_ev.set_defaults(func=_cmd_evaluate)

    p_in = sub.add_parser("infer", help="Run YOLO26 inference on image/video/RTSP")
    p_in.add_argument("--model", type=Path, required=True, help="Path to model weights file (.pt for YOLO, .pth for RF-DETR)")
    p_in.add_argument("--source", type=str, required=True, help="Image path, video path, or RTSP URL")
    p_in.add_argument("--output", type=str, default=None, help="Output video path (optional; if not provided, only JSON output)")
    p_in.add_argument("--conf", type=float, default=0.70, help="Confidence threshold")
    p_in.add_argument("--device", type=str, default="cuda", help="Inference device (cuda/cpu/0)")
    p_in.add_argument("--model-type", "--model_type", type=str, default="auto", choices=["auto", "yolo", "rfdetr"], help="Model type (auto-detect, yolo, or rfdetr)")
    p_in.set_defaults(func=_cmd_infer)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
