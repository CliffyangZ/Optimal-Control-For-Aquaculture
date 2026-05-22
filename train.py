"""CLI entry point for the YOLO Fish & Feed Detection pipeline.

Subcommands:
    download   Download FishTracker_v2 from Roboflow and generate 5-fold splits
    train      Train YOLO26s across folds with wandb logging
    evaluate   Aggregate cross-fold metrics and apply the quality gate
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
        fold_yamls = dataset.generate_kfold_splits()
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(f"[download] ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"[download] Dataset cached: {cache_dir} "
        f"({summary.image_count} images, {len(summary.classes)} classes)"
    )
    print(
        f"[download] 5-fold splits generated: {fold_yamls[0].parent} "
        f"({len(fold_yamls)} fold YAML configs)"
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from core.train.evaluator import CrossFoldEvaluator

    config = load_training_config(args.config)
    # Use --runs-dir if provided, otherwise derive from config.model.output_dir
    if args.runs_dir is not None:
        runs_dir = Path(args.runs_dir)
    else:
        runs_dir = Path(config.model.output_dir)

    # Discover fold best.pt paths: runs_dir/fold-{k}_*/weights/best.pt
    best_pts: dict[int, Path] = {}
    for weights_dir in sorted(runs_dir.glob("fold-*/weights")):
        run_name = weights_dir.parent.name  # fold-0_20260517_1430
        try:
            k = int(run_name.split("-")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        bp = weights_dir / "best.pt"
        if bp.exists():
            best_pts[k] = bp

    if not best_pts:
        print(
            f"[evaluate] ERROR: no fold checkpoints found in {runs_dir}. "
            f"Run `python train.py train` first.",
            file=sys.stderr,
        )
        return 1

    evaluator = CrossFoldEvaluator(config.evaluation)
    fold_metrics_list = []

    # Determine fold YAML paths
    fold_output_dir = Path(config.dataset.fold_output_dir)
    for k in sorted(best_pts):
        model_path = best_pts[k]
        val_yaml = fold_output_dir / f"fold_{k}.yaml"
        if not val_yaml.exists():
            print(
                f"[evaluate] WARNING: fold YAML not found: {val_yaml}; skipping fold {k}",
                file=sys.stderr,
            )
            continue
        print(f"[evaluate] Evaluating fold {k} — {model_path}", flush=True)
        fm = evaluator.evaluate_fold(k, model_path, val_yaml)
        fold_metrics_list.append(fm)

    report = evaluator.aggregate(fold_metrics_list)
    report_path = runs_dir.parent / "cv_report.json"
    evaluator.write_report(report, report_path)
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    from core.infer.detector import YOLODetector
    import json as _json
    import cv2

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[infer] ERROR: model file not found: {model_path}", file=sys.stderr)
        return 1

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
            result = detector.detect(frame)

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
    from core.train.trainer import FoldTrainer

    config = load_training_config(args.config)
    setup_wandb_mode(config.wandb, disabled=args.no_wandb)

    trainer = FoldTrainer(config, project_root=Path.cwd())
    try:
        if args.fold is not None:
            fold_yaml = Path(config.dataset.fold_output_dir) / f"fold_{args.fold}.yaml"
            if not fold_yaml.exists():
                print(
                    f"[train] ERROR: fold YAML not found: {fold_yaml}. "
                    f"Run `python train.py download` first.",
                    file=sys.stderr,
                )
                return 1
            trainer.train_fold(args.fold, fold_yaml)
        else:
            trainer.train_all_folds()
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(f"[train] ERROR: {exc}", file=sys.stderr)
        return 2

    print("[train] All folds complete. Run `python train.py evaluate` for cross-fold report.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train.py", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download dataset + generate 5-fold splits")
    p_dl.add_argument("--config", type=Path, default=Path("config/training.yaml"))
    p_dl.set_defaults(func=_cmd_download)

    p_tr = sub.add_parser("train", help="Train YOLO26s across folds with wandb")
    p_tr.add_argument("--config", type=Path, default=Path("config/training.yaml"))
    p_tr.add_argument(
        "--fold",
        type=int,
        choices=range(0, 5),
        default=None,
        help="Run a single fold (0–4); omit to run all 5",
    )
    p_tr.add_argument(
        "--no-wandb", action="store_true", help="Disable wandb logging entirely"
    )
    p_tr.set_defaults(func=_cmd_train)

    p_ev = sub.add_parser("evaluate", help="Aggregate cross-fold metrics + quality gate")
    p_ev.add_argument("--config", type=Path, default=Path("config/training.yaml"))
    p_ev.add_argument("--runs-dir", type=Path, default=None, help="Override output directory (default: from config)")
    p_ev.set_defaults(func=_cmd_evaluate)

    p_in = sub.add_parser("infer", help="Run YOLO26 inference on image/video/RTSP")
    p_in.add_argument("--model", type=Path, required=True, help="Path to .pt weights file")
    p_in.add_argument("--source", type=str, required=True, help="Image path, video path, or RTSP URL")
    p_in.add_argument("--output", type=str, default=None, help="Output video path (optional; if not provided, only JSON output)")
    p_in.add_argument("--conf", type=float, default=0.70, help="Confidence threshold")
    p_in.add_argument("--device", type=str, default="cuda", help="Inference device (cuda/cpu/0)")
    p_in.set_defaults(func=_cmd_infer)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
