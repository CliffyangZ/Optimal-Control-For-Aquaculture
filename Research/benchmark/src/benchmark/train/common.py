"""Shared run plumbing for all trainers: record lifecycle, hashes, tracking, seeding."""
from __future__ import annotations

import os
import random
import traceback
from pathlib import Path
from typing import Any, Callable

from ..config import REPO_ROOT, Config, ConfigError, base_run_id, run_seed
from ..data.class_map import class_map_hash
from ..data.split import load_manifest, manifest_path
from ..records import (config_hash, git_commit, new_record, read_record, set_status,
                       write_record)


class TrainingFailed(RuntimeError):
    """Training crashed; run record has status=failed (CLI exit code 4)."""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def micro_batch(cfg: Config, run_id: str, override: int | None = None) -> tuple[int, int]:
    """Return (micro_batch, grad_accum) so micro*accum == effective_batch."""
    spec = cfg.run_spec(run_id)
    eff = int(cfg["budget"]["effective_batch"])
    mb = override or int(cfg["budget"]["micro_batch"][spec["family"]][spec["size"]])
    mb = min(mb, eff)
    if eff % mb:
        raise ValueError(f"effective_batch {eff} not divisible by micro_batch {mb}")
    return mb, eff // mb


def wandb_settings(cfg: Config) -> dict[str, Any]:
    return cfg["tracking"]["wandb"]


def wandb_preflight(cfg: Config) -> None:
    """Fail fast (before any run record is written) if W&B tracking is required but unusable."""
    w = wandb_settings(cfg)
    if not w["enabled"]:
        return
    try:
        import wandb
    except ImportError as e:
        raise ConfigError("tracking.wandb.enabled is true but `wandb` is not installed") from e
    if not (os.environ.get("WANDB_API_KEY") or wandb.api.api_key):
        raise ConfigError(
            "W&B tracking is enabled but you are not logged in: run `wandb login` "
            "(or export WANDB_API_KEY). Set tracking.wandb.enabled: false to train without it."
        )


def start_wandb(cfg: Config, run_id: str, rec: dict[str, Any], extra: dict[str, Any] | None = None):
    """Start the single W&B run for this benchmark run (same project/name for all families)."""
    w = wandb_settings(cfg)
    if not w["enabled"]:
        return None
    import wandb
    config = {"run_id": run_id, "family": rec["family"], "size": rec["size"], "seed": rec["seed"],
              "budget": rec["budget"], "micro_batch": rec["micro_batch"],
              "grad_accum_steps": rec["grad_accum_steps"], "imgsz": rec["input_size"],
              "split_hash": rec["split_hash"], "class_map_hash": rec["class_map_hash"],
              "config_hash": rec["config_hash"], "eval_commit": rec["eval_commit"]}
    config.update(extra or {})
    return wandb.init(project=w["project"], entity=w.get("entity"), name=run_id,
                      group=rec["family"], tags=[rec["family"], rec["size"], f"seed{rec['seed']}"],
                      config=config, reinit=True)


def pretrained_path(cfg: Config, run_id: str) -> Path:
    p = REPO_ROOT / cfg.run_spec(run_id)["pretrained"]
    if not p.exists():
        raise FileNotFoundError(f"pretrained weights missing: {p} (run scripts/download_pretrained.py)")
    return p


def init_record(cfg: Config, run_id: str, mb: int, accum: int, input_size: int) -> dict[str, Any]:
    spec = cfg.run_spec(run_id)
    seed = run_seed(run_id, cfg.seed)
    manifest = load_manifest(cfg)
    rec = new_record(
        run_id, spec["family"], spec["size"], seed,
        input_size=input_size,
        budget={"max_epochs": int(cfg["budget"]["max_epochs"]),
                "patience": int(cfg["budget"]["patience"]),
                "effective_batch": int(cfg["budget"]["effective_batch"])},
        eval_commit=git_commit(),
        config_hash=config_hash(cfg.snapshot()),
        class_map_hash=class_map_hash(cfg),
        split_hash=manifest["split_hash"],
    )
    rec["micro_batch"] = mb
    rec["grad_accum_steps"] = accum
    rec["pretrained_path"] = spec["pretrained"]
    return rec


def run_training(cfg: Config, run_id: str, trainer: Callable[..., dict[str, Any]],
                 input_size: int | None = None, micro_batch_override: int | None = None,
                 force: bool = False) -> dict[str, Any]:
    """Run `trainer(cfg, run_id, run_dir, manifest, mb, accum, rec) -> {weights_path, wandb_run_id?, input_size?}`.

    Writes run-record.json through pending -> trained (or failed). A failed run is recorded,
    never silently replaced (spec edge case).
    """
    if not manifest_path(cfg).exists():
        raise FileNotFoundError("split manifest missing; run the `split` stage first")
    run_dir = cfg.run_dir(run_id)
    existing = run_dir / "run-record.json"
    if existing.exists() and not force:
        status = read_record(run_dir)["status"]
        if status in ("trained", "evaluated"):
            raise FileExistsError(f"{run_id} already {status}; pass --force to retrain")
    wandb_preflight(cfg)
    mb, accum = micro_batch(cfg, run_id, micro_batch_override)
    rec = init_record(cfg, run_id, mb, accum, input_size or int(cfg["imgsz"]))
    write_record(run_dir, rec)
    manifest = load_manifest(cfg)
    seed_everything(rec["seed"])
    try:
        out = trainer(cfg, run_id, run_dir, manifest, mb, accum, rec)
        rec["weights_path"] = str(Path(out["weights_path"]).relative_to(REPO_ROOT)) \
            if Path(out["weights_path"]).is_absolute() else str(out["weights_path"])
        if out.get("wandb_run_id"):
            rec["wandb_run_id"] = out["wandb_run_id"]
        if out.get("input_size"):
            rec["input_size"] = int(out["input_size"])
        if out.get("notes"):
            rec["notes"] = out["notes"]
        rec = set_status(rec, "trained")
    except Exception as e:  # noqa: BLE001
        rec = set_status(rec, "failed", f"{type(e).__name__}: {e}")
        write_record(run_dir, rec)
        (run_dir / "train_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise TrainingFailed(f"{run_id}: {rec['failure_reason']}") from e
    write_record(run_dir, rec)
    return rec


def family_of(run_id: str) -> str:
    return {"Y": "yolo26", "R": "rfdetr", "D": "dfine"}[base_run_id(run_id)[0]]
