"""Custom Ultralytics callbacks and wandb mode setup for the training pipeline.

The Ultralytics built-in wandb callback handles per-epoch metrics, hyperparameters,
and the `best.pt` model artifact upload. This module adds:

- `make_val_image_logger(N)`: logs `val_batch*_pred.jpg` to wandb every N epochs.
- `setup_wandb_mode(...)`: probes wandb reachability and falls back to offline mode
  transparently; supports a `disabled` switch wired to `--no-wandb`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

from core.config import WandbConfig


log = logging.getLogger(__name__)


def make_val_image_logger(log_every_n: int = 10) -> Callable[[object], None]:
    """Build an `on_fit_epoch_end` callback that uploads val prediction images.

    The callback fires every `log_every_n` epochs (1-indexed: epochs 10, 20, ...)
    and uploads every `val_batch*_pred.jpg` written by the Ultralytics validator
    to `trainer.save_dir` as a list of `wandb.Image`.

    The callback is a no-op when there is no active `wandb.run`, so it remains
    safe under `WANDB_MODE=disabled`.
    """
    if log_every_n < 1:
        raise ValueError(f"log_every_n must be >= 1, got {log_every_n}")

    def log_val_images(trainer: object) -> None:
        epoch = getattr(trainer, "epoch", None)
        save_dir = getattr(trainer, "save_dir", None)
        if epoch is None or save_dir is None:
            return
        if (epoch + 1) % log_every_n != 0:
            return

        try:
            import wandb
        except ImportError:
            return

        if not getattr(wandb, "run", None):
            return

        imgs = sorted(Path(save_dir).glob("val_batch*_pred.jpg"))
        if not imgs:
            return

        wandb.run.log(
            {"val_predictions": [wandb.Image(str(p)) for p in imgs]},
            step=epoch + 1,
        )

    return log_val_images


def setup_wandb_mode(config: WandbConfig, *, disabled: bool = False) -> str:
    """Configure WANDB_MODE before Ultralytics fires `wandb.init()`.

    Returns the resolved mode (`"disabled"`, `"online"`, or `"offline"`).

    - `disabled=True` (via `--no-wandb`): set `WANDB_MODE=disabled`, skip probe.
    - Otherwise: attempt a 5s `wandb.login()` probe. On failure, fall back to
      `WANDB_MODE=offline` and log instructions for syncing later.
    """
    if disabled:
        os.environ["WANDB_MODE"] = "disabled"
        return "disabled"

    # Set the entity from config so Ultralytics' internal wandb.init() picks it up.
    if config.entity:
        os.environ.setdefault("WANDB_ENTITY", config.entity)

    try:
        import wandb

        wandb.login(timeout=5)
    except Exception as exc:  # noqa: BLE001 — any failure (network, missing key, etc.)
        os.environ["WANDB_MODE"] = "offline"
        log.warning(
            "wandb unreachable (%s): running in offline mode. "
            "Sync later with: wandb sync <run_dir>/wandb/latest-run/",
            exc,
        )
        return "offline"

    return "online"
