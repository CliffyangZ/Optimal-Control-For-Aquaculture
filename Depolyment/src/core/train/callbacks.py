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


def _scalar(value: object) -> int | float | None:
    """Return a wandb-safe scalar from common numeric/tensor values."""
    if hasattr(value, "item"):
        try:
            value = value.item()  # type: ignore[assignment]
        except Exception:  # noqa: BLE001 - tolerate non-scalar tensor-like values
            return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    return None


def _add_scalar_metrics(
    payload: dict[str, int | float], metrics: dict[object, object] | None
) -> None:
    if not isinstance(metrics, dict):
        return
    for key, value in metrics.items():
        scalar = _scalar(value)
        if scalar is not None:
            payload[str(key)] = scalar


def make_wandb_metrics_logger() -> Callable[[object], None]:
    """Build an `on_fit_epoch_end` callback that logs Ultralytics scalars to wandb.

    This is intentionally small and defensive: Ultralytics exposes validation
    metrics on `trainer.metrics`, train losses via `trainer.label_loss_items`,
    and learning rates on `trainer.lr`. Its built-in wandb integration may be
    disabled in user settings, so logging these ourselves ensures an active run
    receives the loss and metric curves every epoch.
    """

    def log_metrics(trainer: object) -> None:
        try:
            import wandb
        except ImportError:
            return

        if not getattr(wandb, "run", None):
            return

        payload: dict[str, int | float] = {}
        epoch = getattr(trainer, "epoch", None)
        if isinstance(epoch, int):
            payload["epoch"] = epoch + 1

        label_loss_items = getattr(trainer, "label_loss_items", None)
        tloss = getattr(trainer, "tloss", None)
        if callable(label_loss_items) and tloss is not None:
            try:
                _add_scalar_metrics(payload, label_loss_items(tloss, prefix="train"))
            except Exception:  # noqa: BLE001 - training should not fail because logging did
                log.debug("Unable to collect Ultralytics train losses for wandb", exc_info=True)

        _add_scalar_metrics(payload, getattr(trainer, "metrics", None))
        _add_scalar_metrics(payload, getattr(trainer, "lr", None))

        if len(payload) <= 1 and "epoch" in payload:
            return

        step = (epoch + 1) if isinstance(epoch, int) else None
        wandb.run.log(payload, step=step)

    return log_metrics


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

    # Set wandb env vars so both our explicit run and any Ultralytics integration
    # resolve to the configured project/entity.
    os.environ.setdefault("WANDB_DIR", "results")
    os.environ["WANDB_PROJECT"] = config.project
    if config.entity:
        os.environ["WANDB_ENTITY"] = config.entity

    try:
        import wandb

        wandb.require("core")
        wandb.login(timeout=10)
    except Exception as exc:  # noqa: BLE001 — any failure (network, missing key, etc.)
        os.environ["WANDB_MODE"] = "offline"
        msg = (
            f"wandb unreachable ({exc}): running in offline mode. No cloud logging. "
            "Run `wandb login --relogin` with a valid key from https://wandb.ai/authorize, "
            "then sync with: wandb sync Results/wandb/latest-run/"
        )
        log.warning(msg)
        print(f"[wandb] {msg}", flush=True)
        return "offline"

    os.environ["WANDB_MODE"] = "online"
    print("[wandb] authenticated — online mode", flush=True)
    return "online"
