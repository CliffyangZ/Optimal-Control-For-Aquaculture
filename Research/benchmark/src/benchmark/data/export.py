"""Manifest -> trainer formats (YOLO txt, COCO JSON) and evaluation ground truth."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import cv2

from ..config import REPO_ROOT, Config
from .class_map import class_index, remap_id

CAT_OFFSET = 1  # COCO category id = unified class index + 1


def image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    h, w = img.shape[:2]
    return w, h


def read_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Return [(source_class_id, cx, cy, w, h)] normalised; polygons become boxes."""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:
            cx, cy, w, h = vals
        else:  # polygon x1 y1 x2 y2 ...
            xs, ys = vals[0::2], vals[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        out.append((cid, cx, cy, w, h))
    return out


def entry_boxes(cfg: Config, entry: dict) -> list[tuple[int, float, float, float, float]]:
    """[(unified_class_id, cx, cy, w, h)] normalised for one manifest entry."""
    boxes = read_yolo_boxes(REPO_ROOT / entry["label"])
    return [(remap_id(cfg, entry["source_dataset"], c), cx, cy, w, h) for c, cx, cy, w, h in boxes]


def gt_coco(cfg: Config, manifest: dict, partition: str) -> dict:
    """COCO-style ground truth for one partition. image id = index in partition."""
    names = cfg.classes
    images, anns = [], []
    aid = 1
    for i, e in enumerate(manifest["partitions"][partition]):
        path = REPO_ROOT / e["path"]
        w, h = image_size(path)
        images.append({"id": i, "file_name": e["path"], "width": w, "height": h})
        for cid, cx, cy, bw, bh in entry_boxes(cfg, e):
            x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
            bw_p, bh_p = bw * w, bh * h
            anns.append({
                "id": aid, "image_id": i, "category_id": cid + CAT_OFFSET,
                "bbox": [x, y, bw_p, bh_p], "area": bw_p * bh_p, "iscrowd": 0,
            })
            aid += 1
    cats = [{"id": i + CAT_OFFSET, "name": n} for i, n in enumerate(names)]
    return {"images": images, "annotations": anns, "categories": cats}


def write_gt(cfg: Config, manifest: dict, partitions=("train", "val", "heldout_real")) -> dict[str, Path]:
    out_dir = cfg.results_root / "splits" / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for part in partitions:
        p = out_dir / f"gt_{part}.json"
        p.write_text(json.dumps(gt_coco(cfg, manifest, part)), encoding="utf-8")
        paths[part] = p
    return paths


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_yolo(cfg: Config, manifest: dict, out_dir: Path | None = None) -> Path:
    """Write YOLO-layout dataset (train/val) with remapped labels; returns data.yaml path."""
    root = out_dir or cfg.results_root / "splits" / "export" / "yolo"
    for part, sub in (("train", "train"), ("val", "val")):
        for e in manifest["partitions"][part]:
            src = REPO_ROOT / e["path"]
            name = f"{e['sha256'][:12]}{src.suffix.lower()}"
            _link(src, root / sub / "images" / name)
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in entry_boxes(cfg, e)]
            lab = root / sub / "labels" / f"{e['sha256'][:12]}.txt"
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_text("\n".join(lines), encoding="utf-8")
    data = {
        "path": str(root), "train": "train/images", "val": "val/images",
        "nc": len(cfg.classes), "names": cfg.classes,
    }
    import yaml
    yp = root / "data.yaml"
    yp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yp


def export_coco_dir(cfg: Config, manifest: dict, out_dir: Path | None = None,
                    cat_offset: int = 0) -> Path:
    """Roboflow-style COCO layout: train/ valid/ each with images + _annotations.coco.json."""
    root = out_dir or cfg.results_root / "splits" / "export" / "coco"
    for part, sub in (("train", "train"), ("val", "valid")):
        gt = gt_coco(cfg, manifest, part)
        for img in gt["images"]:
            src = REPO_ROOT / img["file_name"]
            name = f"{Path(img['file_name']).stem[:40]}_{img['id']}{src.suffix.lower()}"
            _link(src, root / sub / name)
            img["file_name"] = name
        for a in gt["annotations"]:
            a["category_id"] += cat_offset - CAT_OFFSET
        for c in gt["categories"]:
            c["id"] += cat_offset - CAT_OFFSET
        (root / sub).mkdir(parents=True, exist_ok=True)
        (root / sub / "_annotations.coco.json").write_text(json.dumps(gt), encoding="utf-8")
    return root


def class_index_map(cfg: Config) -> dict[str, int]:
    return class_index(cfg)
