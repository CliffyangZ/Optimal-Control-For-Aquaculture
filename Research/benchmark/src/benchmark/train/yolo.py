"""YOLO26 training (Ultralytics). Augmentation/optimiser block comes from src/config/training.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config import REPO_ROOT, Config
from ..data.export import export_yolo
from .common import pretrained_path, start_wandb, wandb_settings

# keys of training.yaml `model:` that Ultralytics understands (others, e.g. auto_weight, are ignored)
YOLO_PASSTHROUGH = (
    "mosaic", "close_mosaic", "mixup", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "warmup_momentum", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
    "scale", "flipud", "fliplr", "perspective", "erasing", "copy_paste",
)


def load_yolo_hyp(cfg: Config) -> dict[str, Any]:
    raw = yaml.safe_load((REPO_ROOT / cfg["yolo_training_yaml"]).read_text(encoding="utf-8"))["model"]
    return {k: raw[k] for k in YOLO_PASSTHROUGH if k in raw}


def build_train_args(cfg: Config, run_id: str, data_yaml: Path, run_dir: Path,
                     mb: int, seed: int) -> dict[str, Any]:
    b = cfg["budget"]
    args = dict(
        data=str(data_yaml),
        epochs=int(b["max_epochs"]),
        patience=int(b["patience"]),
        imgsz=int(cfg["imgsz"]),
        batch=mb,
        nbs=int(b["effective_batch"]),   # Ultralytics accumulates to reach nbs => effective batch
        seed=seed,
        deterministic=True,
        project=str(run_dir),
        name="ultralytics",
        exist_ok=True,
        workers=4,
        pretrained=True,
        plots=True,
        device=0,
    )
    args.update(load_yolo_hyp(cfg))
    return args


def train(cfg: Config, run_id: str, run_dir: Path, manifest: dict, mb: int, accum: int,
          rec: dict[str, Any]) -> dict[str, Any]:
    from ultralytics import YOLO, settings

    w = wandb_settings(cfg)
    # mlflow is pulled in by rfdetr[loggers]; its file store aborts training
    settings.update({"wandb": bool(w["enabled"]), "mlflow": False})
    # Pre-start the run so Ultralytics' callback reuses it (otherwise it names the project after the output path)
    wb = start_wandb(cfg, run_id, rec)
    data_yaml = export_yolo(cfg, manifest)
    args = build_train_args(cfg, run_id, data_yaml, run_dir, mb, rec["seed"])
    model = YOLO(str(pretrained_path(cfg, run_id)))
    # NOTE: Ultralytics keeps `nbs` fixed; with batch=mb it accumulates round(nbs/mb) steps.
    try:
        model.train(**args)
    finally:
        wid = wb.id if wb else None
        if wb:
            wb.finish()
    best = run_dir / "ultralytics" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"expected best checkpoint at {best}")
    return {"weights_path": best, "input_size": args["imgsz"], "wandb_run_id": wid}
