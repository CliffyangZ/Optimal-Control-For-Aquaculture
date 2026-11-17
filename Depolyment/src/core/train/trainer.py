"""YOLO26 training with wandb logging.

The Ultralytics built-in wandb callback handles per-epoch metrics and the
auto `best.pt` artifact upload. We additionally upload the final `best.pt`
as a wandb artifact for discoverability.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from core.config import TrainingConfig
from core.train.callbacks import make_val_image_logger, make_wandb_metrics_logger


log = logging.getLogger(__name__)


class Trainer:
    """Train a single YOLO26 model with wandb logging."""

    def __init__(self, config: TrainingConfig, project_root: Path | None = None):
        self.config = config
        self.project_root = (project_root or Path.cwd()).resolve()

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M")

    def _wandb_disabled(self) -> bool:
        return os.environ.get("WANDB_MODE", "").lower() == "disabled"

    def _start_wandb_run(self, run_name: str) -> tuple[object | None, bool]:
        """Start a wandb run before Ultralytics training begins."""
        if self._wandb_disabled():
            return None, False

        try:
            import wandb
        except ImportError:
            print("[wandb] package not installed; skipping wandb logging", flush=True)
            return None, False

        active_run = getattr(wandb, "run", None)
        if active_run:
            return active_run, False

        run = wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity or None,
            name=run_name,
            dir="results",
            tags=["train"],
            config={
                "weights": self.config.model.weights,
                "variant": self.config.model.variant,
                "epochs": self.config.model.epochs,
                "batch": self.config.model.batch,
                "imgsz": self.config.model.imgsz,
                "seed": self.config.model.seed,
                "dataset": "FishTracker_v2",
            },
        )
        try:
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")
        except Exception:  # noqa: BLE001 - keep training running if wandb chart setup fails
            log.debug("Unable to define wandb epoch metric", exc_info=True)
        return run, True

    def _finish_wandb_run(self, run: object | None, should_finish: bool) -> None:
        """Finish only the run this trainer explicitly started."""
        if run is None or not should_finish:
            return
        try:
            import wandb
        except ImportError:
            return
        if getattr(wandb, "run", None) is run:
            wandb.finish()

    def train(self, data_yaml: Path) -> Path:
        """Train the model and return its Ultralytics `save_dir`."""
        from ultralytics import YOLO  # local import: ml group only

        timestamp = self._timestamp()
        run_name = f"run_{timestamp}"
        print(f"[train] Starting training → wandb run: {run_name}")

        wandb_run, should_finish_wandb = self._start_wandb_run(run_name)
        try:
            model = YOLO(self.config.model.weights)
            model.add_callback("on_fit_epoch_end", make_wandb_metrics_logger())
            model.add_callback(
                "on_fit_epoch_end",
                make_val_image_logger(self.config.wandb.log_val_images_every_n),
            )

            results = model.train(
                data=str(data_yaml),
                project=str(self.project_root / self.config.model.output_dir),
                name=run_name,
                epochs=self.config.model.epochs,
                batch=self.config.model.batch,
                imgsz=self.config.model.imgsz,
                workers=self.config.model.workers,
                mosaic=self.config.model.mosaic,
                close_mosaic=self.config.model.close_mosaic,
                mixup=self.config.model.mixup,
                patience=self.config.model.patience,
                seed=self.config.model.seed,
                hsv_h=self.config.model.hsv_h,
                hsv_s=self.config.model.hsv_s,
                hsv_v=self.config.model.hsv_v,
                degrees=self.config.model.degrees,
                translate=self.config.model.translate,
                scale=self.config.model.scale,
                flipud=self.config.model.flipud,
                fliplr=self.config.model.fliplr,
            )

            save_dir = Path(
                getattr(results, "save_dir", None)
                or getattr(model.trainer, "save_dir", None)
                or (self.project_root / self.config.model.output_dir / run_name)
            )
            best_pt = save_dir / "weights" / "best.pt"

            print(
                f"[train] Training complete: best.pt → {best_pt}"
                + (f"  (wandb run: {run_name})" if not self._wandb_disabled() else "")
            )

            self.upload_artifact(save_dir)
            return save_dir
        finally:
            self._finish_wandb_run(wandb_run, should_finish_wandb)

    def upload_artifact(self, save_dir: Path) -> None:
        """Log `best.pt` as a wandb artifact (skipped if wandb disabled)."""
        if self._wandb_disabled():
            return

        try:
            import wandb
        except ImportError:
            return

        if not getattr(wandb, "run", None):
            return

        best_pt = save_dir / "weights" / "best.pt"
        if not best_pt.exists():
            log.warning("No best.pt found at %s; skipping artifact upload", best_pt)
            return

        artifact = wandb.Artifact(
            name="aquaculture-best",
            type="model",
            metadata={
                "dataset": "FishTracker_v2",
                "epochs": self.config.model.epochs,
            },
        )
        artifact.add_file(str(best_pt))
        wandb.run.log_artifact(artifact, aliases=["best"])
