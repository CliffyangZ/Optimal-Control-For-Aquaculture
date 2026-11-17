"""Evaluation-set image lists and the adapter dispatch shared by all families."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..config import REPO_ROOT, Config, family_of_run
from ..data.export import gt_coco
from ..data.split import load_manifest

Predictions = list[dict[str, Any]]


def eval_set_images(cfg: Config, eval_set: str) -> list[tuple[int, Path]]:
    """[(image_id, path)]; image_id = index within the partition (same ids as the GT file)."""
    if eval_set == "synthetic":
        raise NotImplementedError("synthetic eval set is not implemented yet (task T043)")
    manifest = load_manifest(cfg)
    return [(i, REPO_ROOT / e["path"]) for i, e in enumerate(manifest["partitions"][eval_set])]


def eval_set_gt(cfg: Config, eval_set: str) -> dict[str, Any]:
    if eval_set == "synthetic":
        raise NotImplementedError("synthetic eval set is not implemented yet (task T043)")
    return gt_coco(cfg, load_manifest(cfg), eval_set)


def get_adapter(run_id: str) -> Callable[..., Predictions]:
    fam = family_of_run(run_id)
    if fam == "yolo26":
        from .adapter_yolo import predict
    elif fam == "rfdetr":
        from .adapter_rfdetr import predict
    else:
        from .adapter_dfine import predict
    return predict


def resolve_weights(rec: dict[str, Any]) -> Path:
    if not rec.get("weights_path"):
        raise FileNotFoundError(f"{rec['run_id']}: no weights_path in run record (train first)")
    p = Path(rec["weights_path"])
    p = p if p.is_absolute() else REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"weights not found: {p}")
    return p
