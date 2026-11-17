"""RF-DETR training via the `rfdetr` package. Uses each size's official resolution (declared in record)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Config
from ..data.export import export_coco_dir
from .common import pretrained_path, wandb_settings

SIZE_CLASS = {"n": "RFDETRNano", "s": "RFDETRSmall", "m": "RFDETRMedium"}
# official input resolution per size (rfdetr 1.7 defaults); recorded in the run record
OFFICIAL_RESOLUTION = {"n": 384, "s": 512, "m": 576}


def build_train_kwargs(cfg: Config, run_id: str, dataset_dir: Path, run_dir: Path,
                       mb: int, accum: int, seed: int) -> dict[str, Any]:
    b = cfg["budget"]
    w = wandb_settings(cfg)
    kw = dict(
        dataset_dir=str(dataset_dir),
        output_dir=str(run_dir / "rfdetr"),
        epochs=int(b["max_epochs"]),
        batch_size=mb,
        grad_accum_steps=accum,
        early_stopping=True,
        early_stopping_patience=int(b["patience"]),
        seed=seed,
        class_names=cfg.classes,
        num_workers=4,
        wandb=bool(w["enabled"]),
    )
    if w["enabled"]:
        kw["project"] = w["project"]
        kw["run"] = run_id
    return kw


def find_best_checkpoint(out_dir: Path) -> Path:
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth",
                 "checkpoint_best_regular.pth"):
        p = out_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no best checkpoint found in {out_dir}")


def train(cfg: Config, run_id: str, run_dir: Path, manifest: dict, mb: int, accum: int,
          rec: dict[str, Any]) -> dict[str, Any]:
    import rfdetr

    size = cfg.run_spec(run_id)["size"]
    # COCO ids 1..N (Roboflow convention); rfdetr derives the class count from the categories
    dataset_dir = export_coco_dir(cfg, manifest, cat_offset=1)
    model_cls = getattr(rfdetr, SIZE_CLASS[size])
    model = model_cls(pretrain_weights=str(pretrained_path(cfg, run_id)), num_classes=len(cfg.classes))
    kwargs = build_train_kwargs(cfg, run_id, dataset_dir, run_dir, mb, accum, rec["seed"])
    model.train(**kwargs)
    try:  # rfdetr starts the W&B run itself; read its id for the run record
        import wandb
        wid = wandb.run.id if wandb.run else None
        if wandb.run:
            wandb.run.finish()
    except Exception:  # noqa: BLE001
        wid = None
    best = find_best_checkpoint(run_dir / "rfdetr")
    return {
        "weights_path": best,
        "input_size": OFFICIAL_RESOLUTION[size],
        "wandb_run_id": wid,
        "notes": f"RF-DETR official resolution {OFFICIAL_RESOLUTION[size]} (not cfg.imgsz)",
    }
