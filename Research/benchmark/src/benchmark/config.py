"""Benchmark configuration loading and validation."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# .../Research/benchmark/src/benchmark/config.py -> repo root is parents[4]
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "Research" / "benchmark" / "configs" / "benchmark.yaml"

REQUIRED_KEYS = {
    "seed", "paths", "classes", "source_class_maps", "kpi_classes", "split",
    "budget", "imgsz", "thresholds", "bootstrap", "profile", "runs", "decision",
    "tracking", "yolo_training_yaml",
}
RUN_FAMILIES = {"yolo26", "rfdetr", "dfine"}
RUN_SIZES = {"n", "s", "m"}
FAMILY_PREFIX = {"Y": "yolo26", "R": "rfdetr", "D": "dfine"}


class ConfigError(ValueError):
    """Invalid benchmark configuration (CLI exit code 2)."""


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def classes(self) -> list[str]:
        return list(self.raw["classes"])

    @property
    def thresholds(self) -> dict[str, float]:
        return self.raw["thresholds"]

    @property
    def results_root(self) -> Path:
        return REPO_ROOT / self.raw["paths"]["results_root"]

    @property
    def datasets_root(self) -> Path:
        return REPO_ROOT / self.raw["paths"]["datasets_root"]

    def run_dir(self, run_id: str) -> Path:
        return self.results_root / "runs" / run_id

    def run_spec(self, run_id: str) -> dict[str, Any]:
        base = base_run_id(run_id)
        if base not in self.raw["runs"]:
            raise ConfigError(f"unknown run '{run_id}'")
        return self.raw["runs"][base]

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw)


def family_of_run(run_id: str) -> str:
    prefix = run_id.split("-")[0]
    if prefix not in FAMILY_PREFIX:
        raise ConfigError(f"bad run id '{run_id}' (expected Y-/R-/D-)")
    return FAMILY_PREFIX[prefix]


def base_run_id(run_id: str) -> str:
    """'Y-s-seed7' -> 'Y-s'."""
    parts = run_id.split("-")
    return "-".join(parts[:2])


def run_seed(run_id: str, default: int) -> int:
    parts = run_id.split("-")
    if len(parts) == 3 and parts[2].startswith("seed"):
        return int(parts[2][4:])
    return default


def validate(raw: dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - raw.keys()
    unknown = raw.keys() - REQUIRED_KEYS
    if missing:
        raise ConfigError(f"missing config keys: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")
    for kc in raw["kpi_classes"]:
        if kc not in raw["classes"]:
            raise ConfigError(f"kpi class '{kc}' not in classes")
    for src, m in raw["source_class_maps"].items():
        for cid, name in m.items():
            if name not in raw["classes"]:
                raise ConfigError(f"{src}: class '{name}' not in classes")
    frac = raw["split"]["fractions"]
    if abs(sum(frac.values()) - 1.0) > 1e-6:
        raise ConfigError("split fractions must sum to 1")
    for rid, spec in raw["runs"].items():
        if spec.get("family") not in RUN_FAMILIES or spec.get("size") not in RUN_SIZES:
            raise ConfigError(f"run {rid}: bad family/size")
    t = raw["thresholds"]
    for k in ("conf", "ap_conf_floor", "max_det", "nms_iou", "eval_iou"):
        if k not in t:
            raise ConfigError(f"thresholds.{k} missing")


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"config not found: {p}") from e
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    validate(raw)
    return Config(raw)
