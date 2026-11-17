"""Unified class map (spec FR-002): pellet->feed, feeding-fish->fish; debris/lumpfish kept."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import Config
from ..records import sha256_json


def class_names(cfg: Config) -> list[str]:
    return cfg.classes


def class_index(cfg: Config) -> dict[str, int]:
    return {n: i for i, n in enumerate(cfg.classes)}


def remap_id(cfg: Config, source: str, source_id: int) -> int:
    """Map a source-dataset class id to the unified class id."""
    m = cfg["source_class_maps"][source]
    if source_id not in m:
        raise KeyError(f"{source}: class id {source_id} not in class map")
    return class_index(cfg)[m[source_id]]


def snapshot(cfg: Config) -> dict:
    return {
        "classes": cfg.classes,
        "source_class_maps": {s: {str(k): v for k, v in m.items()}
                              for s, m in cfg["source_class_maps"].items()},
        "kpi_classes": cfg["kpi_classes"],
    }


def class_map_hash(cfg: Config) -> str:
    return sha256_json(snapshot(cfg))


def write_snapshot(cfg: Config) -> Path:
    out = cfg.results_root / "splits" / "class_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    snap = snapshot(cfg)
    snap["hash"] = class_map_hash(cfg)
    out.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
