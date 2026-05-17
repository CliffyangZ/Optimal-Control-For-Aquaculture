"""Training/inference configuration loader.

Reads `config/training.yaml` into typed dataclasses and loads `.env` for
API credentials (Roboflow, wandb).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    roboflow_workspace: str
    roboflow_project: str
    roboflow_version: int
    roboflow_format: str
    local_cache_dir: Path
    n_folds: int
    fold_output_dir: Path


@dataclass
class ModelConfig:
    weights: str
    variant: str
    epochs: int
    batch: int
    imgsz: int
    workers: int
    mosaic: float
    close_mosaic: int
    mixup: float
    patience: int
    seed: int


@dataclass
class WandbConfig:
    project: str
    entity: str
    log_val_images_every_n: int


@dataclass
class ClassThreshold:
    map50: float | None = None
    recall: float | None = None
    precision: float | None = None


@dataclass
class EvaluationConfig:
    thresholds: dict[str, ClassThreshold] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    dataset: DatasetConfig
    model: ModelConfig
    wandb: WandbConfig
    evaluation: EvaluationConfig


def _load_dotenv(env_path: Path) -> None:
    """Load .env into os.environ. Uses python-dotenv if available, else no-op."""
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        # python-dotenv is an `[ml]` group dep; tolerate its absence so
        # non-training code paths can still import this module.
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def load_training_config(
    path: str | Path = "config/training.yaml",
    *,
    env_path: str | Path = ".env",
) -> TrainingConfig:
    """Parse `config/training.yaml` and load `.env` API keys into the environment."""
    _load_dotenv(Path(env_path))

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    ds = raw["dataset"]
    dataset = DatasetConfig(
        roboflow_workspace=ds["roboflow_workspace"],
        roboflow_project=ds["roboflow_project"],
        roboflow_version=int(ds["roboflow_version"]),
        roboflow_format=ds["roboflow_format"],
        local_cache_dir=Path(ds["local_cache_dir"]),
        n_folds=int(ds["n_folds"]),
        fold_output_dir=Path(ds["fold_output_dir"]),
    )

    m = raw["model"]
    model = ModelConfig(
        weights=m["weights"],
        variant=m["variant"],
        epochs=int(m["epochs"]),
        batch=int(m["batch"]),
        imgsz=int(m["imgsz"]),
        workers=int(m["workers"]),
        mosaic=float(m["mosaic"]),
        close_mosaic=int(m["close_mosaic"]),
        mixup=float(m["mixup"]),
        patience=int(m["patience"]),
        seed=int(m["seed"]),
    )

    w = raw["wandb"]
    wandb_cfg = WandbConfig(
        project=w["project"],
        entity=w.get("entity", ""),
        log_val_images_every_n=int(w["log_val_images_every_n"]),
    )

    raw_eval = raw.get("evaluation", {}) or {}
    thresholds_raw = raw_eval.get("thresholds", {}) or {}
    thresholds: dict[str, ClassThreshold] = {}
    for class_name, t in thresholds_raw.items():
        thresholds[class_name] = ClassThreshold(
            map50=(float(t["map50"]) if "map50" in t else None),
            recall=(float(t["recall"]) if "recall" in t else None),
            precision=(float(t["precision"]) if "precision" in t else None),
        )
    evaluation = EvaluationConfig(thresholds=thresholds)

    return TrainingConfig(
        dataset=dataset, model=model, wandb=wandb_cfg, evaluation=evaluation
    )
