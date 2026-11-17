"""Locked train/val/heldout_real split (spec FR-002, research R8)."""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from ..config import REPO_ROOT, Config
from ..records import sha256_file, sha256_json
from .class_map import class_map_hash, write_snapshot

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
PARTITIONS = ("train", "val", "heldout_real")
# (source name, relative dir, subsets)
SOURCES = {
    "fishtracker_v2": ("fishtracker_v2", ("train", "valid", "test")),
    "feeding_fish": ("feeding_fish/feeding_fish.v2i.yolov8", ("train", "valid", "test")),
}


class LockedArtifactError(RuntimeError):
    """Would overwrite a locked artifact (CLI exit code 3)."""


def manifest_path(cfg: Config) -> Path:
    return cfg.results_root / "splits" / "split_manifest.json"


def group_key(name: str) -> str:
    """Video/source group of a Roboflow image name (used only for stratification)."""
    n = name.split(".rf.")[0]
    n = re.sub(r"_jpg$", "", n)
    return re.sub(r"[-_]?(frame_)?\d+$", "", n)


def _collect(cfg: Config) -> list[dict]:
    entries, seen = [], {}
    for source, (rel, subsets) in SOURCES.items():
        for sub in subsets:
            img_dir = cfg.datasets_root / rel / sub / "images"
            if not img_dir.is_dir():
                continue
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() not in IMG_EXT:
                    continue
                digest = sha256_file(p)
                if digest in seen:  # identical bytes: keep first, disjointness by hash
                    continue
                label = p.parent.parent / "labels" / (p.stem + ".txt")
                e = {
                    "path": str(p.relative_to(REPO_ROOT)),
                    "label": str(label.relative_to(REPO_ROOT)),
                    "sha256": digest,
                    "source_dataset": source,
                    "group": group_key(p.name),
                }
                seen[digest] = e
                entries.append(e)
    return entries


def _allocate(n: int, fractions: dict[str, float]) -> dict[str, int]:
    counts = {k: int(n * v) for k, v in fractions.items()}
    if n >= 3:
        for k in ("val", "heldout_real"):
            counts[k] = max(counts[k], 1)
    counts["train"] = n - counts["val"] - counts["heldout_real"]
    return counts


def make_split(entries: list[dict], fractions: dict[str, float], seed: int) -> dict[str, list[dict]]:
    """Stratified (source, group) image-level split, deterministic for a given seed."""
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for e in entries:
        strata[(e["source_dataset"], e["group"])].append(e)
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {k: [] for k in PARTITIONS}
    for key in sorted(strata):
        items = sorted(strata[key], key=lambda e: e["sha256"])
        rng.shuffle(items)
        c = _allocate(len(items), fractions)
        i = 0
        for part in ("heldout_real", "val", "train"):  # held-out carved out first
            out[part].extend(items[i:i + c[part]])
            i += c[part]
    for part in out:
        out[part].sort(key=lambda e: e["sha256"])
    return out


def build_manifest(cfg: Config) -> dict:
    parts = make_split(_collect(cfg), cfg["split"]["fractions"], int(cfg["split"]["seed"]))
    manifest = {
        "seed": int(cfg["split"]["seed"]),
        "class_map_hash": class_map_hash(cfg),
        "partitions": parts,
    }
    manifest["split_hash"] = sha256_json({k: [e["sha256"] for e in v] for k, v in parts.items()})
    return manifest


def validate_manifest(m: dict, class_ids: set[int] | None = None) -> None:
    seen: dict[str, str] = {}
    for part, items in m["partitions"].items():
        for e in items:
            if e["sha256"] in seen and seen[e["sha256"]] != part:
                raise ValueError(f"image in two partitions: {e['path']}")
            seen[e["sha256"]] = part


def write_split(cfg: Config, force: bool = False) -> Path:
    p = manifest_path(cfg)
    if p.exists() and not force:
        raise LockedArtifactError(f"{p} exists; use --force to overwrite")
    write_snapshot(cfg)
    m = build_manifest(cfg)
    validate_manifest(m)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    return p


def load_manifest(cfg: Config) -> dict:
    return json.loads(manifest_path(cfg).read_text(encoding="utf-8"))
