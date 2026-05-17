"""YOLO26 training loop across 5 folds with wandb logging.

Per research.md §3: `YOLO(weights)` MUST be re-instantiated inside the fold loop
to prevent state bleed between folds. The Ultralytics built-in wandb callback
handles per-epoch metrics and the auto `best.pt` artifact upload; we register a
custom val-image callback and additionally upload a fold-tagged artifact for
discoverability.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from core.config import TrainingConfig
from core.train.callbacks import make_val_image_logger, make_wandb_metrics_logger


log = logging.getLogger(__name__)


class FoldTrainer:
    """Train YOLO26s sequentially across all configured k-fold splits."""

    def __init__(self, config: TrainingConfig, project_root: Path | None = None):
        self.config = config
        self.project_root = (project_root or Path.cwd()).resolve()
        self.fold_output_dir = (self.project_root / config.dataset.fold_output_dir).resolve()

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M")

    def _wandb_disabled(self) -> bool:
        return os.environ.get("WANDB_MODE", "").lower() == "disabled"

    def _start_wandb_run(self, k: int, run_name: str) -> tuple[object | None, bool]:
        """Start a fold-scoped wandb run before Ultralytics training begins."""
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
            tags=["train", f"fold-{k}"],
            config={
                "fold": k,
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

    def train_fold(self, k: int, fold_yaml: Path) -> Path:
        """Train one fold and return its Ultralytics `save_dir`."""
        from ultralytics import YOLO  # local import: ml group only

        timestamp = self._timestamp()
        run_name = f"fold-{k}_{timestamp}"
        print(f"[train] Starting fold {k}/{self.config.dataset.n_folds} → wandb run: {run_name}")

        wandb_run, should_finish_wandb = self._start_wandb_run(k, run_name)
        try:
            model = YOLO(self.config.model.weights)
            model.add_callback("on_fit_epoch_end", make_wandb_metrics_logger())
            model.add_callback(
                "on_fit_epoch_end",
                make_val_image_logger(self.config.wandb.log_val_images_every_n),
            )

            results = model.train(
                data=str(fold_yaml),
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
            )

            save_dir = Path(getattr(results, "save_dir", None) or getattr(model.trainer, "save_dir"))
            best_pt = save_dir / "weights" / "best.pt"

            print(
                f"[train] Fold {k} complete: best.pt → {best_pt}"
                + (f"  (wandb run: {run_name})" if not self._wandb_disabled() else "")
            )

            self.upload_fold_artifact(k, save_dir)
            return save_dir
        finally:
            self._finish_wandb_run(wandb_run, should_finish_wandb)

    def upload_fold_artifact(self, k: int, save_dir: Path) -> None:
        """Log `best.pt` as a fold-tagged wandb artifact (skipped if wandb disabled)."""
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
            name=f"aquaculture-fold-{k}",
            type="model",
            metadata={
                "fold": k,
                "dataset": "FishTracker_v2",
                "epochs": self.config.model.epochs,
            },
        )
        artifact.add_file(str(best_pt))
        wandb.run.log_artifact(artifact, aliases=["best", f"fold-{k}"])

    def train_all_folds(self) -> list[Path]:
        """Iterate every `fold_{i}.yaml` in the fold output directory."""
        fold_yamls = sorted(self.fold_output_dir.glob("fold_*.yaml"))
        if not fold_yamls:
            raise FileNotFoundError(
                f"No fold_*.yaml found in {self.fold_output_dir}. "
                f"Run `python train.py download` first."
            )

        save_dirs: list[Path] = []
        for yaml_path in fold_yamls:
            # Expect "fold_{k}.yaml"
            k = int(yaml_path.stem.split("_")[1])
            save_dirs.append(self.train_fold(k, yaml_path))
        return save_dirs
