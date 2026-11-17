"""D-FINE fine-tuning with Hugging Face transformers (custom loop).

Official-style augmentation is approximated with albumentations (flip, photometric, affine
scale/translate); the choice is declared in the run record. Early stopping uses COCO AP@[.5:.95]
on the locked `val` split (checkpoint selection only; the benchmark metrics come from the
shared evaluator).
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import REPO_ROOT, Config
from ..data.export import CAT_OFFSET, entry_boxes, gt_coco, image_size
from .common import pretrained_path, start_wandb

LR = 1e-4
LR_BACKBONE_MULT = 0.1
WEIGHT_DECAY = 1e-4
CLIP_NORM = 0.1
WARMUP_STEPS = 100
MIN_LR_FACTOR = 0.05
TOPK = 300


def _no_grad(fn):
    import torch
    return torch.no_grad()(fn)


def clip_xywh(x: float, y: float, w: float, h: float, W: int, H: int) -> tuple[float, float, float, float] | None:
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(W), x + w), min(float(H), y + h)
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def lr_factor(step: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to MIN_LR_FACTOR."""
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    prog = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return MIN_LR_FACTOR + (1 - MIN_LR_FACTOR) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def to_normalized_cxcywh(boxes_xywh: np.ndarray, W: int, H: int) -> np.ndarray:
    b = np.asarray(boxes_xywh, dtype=np.float32).reshape(-1, 4)
    out = np.empty_like(b)
    out[:, 0] = (b[:, 0] + b[:, 2] / 2) / W
    out[:, 1] = (b[:, 1] + b[:, 3] / 2) / H
    out[:, 2] = b[:, 2] / W
    out[:, 3] = b[:, 3] / H
    return np.clip(out, 0.0, 1.0)


def decode_predictions(logits, pred_boxes, sizes, topk: int = TOPK):
    """DETR sigmoid-head decode -> per image list of (cls, score, x, y, w, h) in original pixels.

    logits (B,Q,C), pred_boxes (B,Q,4) normalised cxcywh, sizes list[(W,H)].
    """
    import torch

    B, Q, C = logits.shape
    scores = logits.float().sigmoid().reshape(B, Q * C)
    k = min(topk, Q * C)
    top_s, top_i = scores.topk(k, dim=1)
    q_idx, c_idx = top_i // C, top_i % C
    out = []
    for b in range(B):
        W, H = sizes[b]
        box = pred_boxes[b].float()[q_idx[b]]
        x = (box[:, 0] - box[:, 2] / 2) * W
        y = (box[:, 1] - box[:, 3] / 2) * H
        w, h = box[:, 2] * W, box[:, 3] * H
        rows = torch.stack([c_idx[b].float(), top_s[b], x, y, w, h], dim=1).detach().cpu().numpy()
        out.append(rows)
    return out


class DetDataset:
    """Manifest partition -> (pixel_values[0..1], labels dict, meta)."""

    def __init__(self, cfg: Config, manifest: dict, part: str, imgsz: int, train: bool):
        import albumentations as A

        self.imgsz, self.train = imgsz, train
        self.items = []
        for e in manifest["partitions"][part]:
            path = REPO_ROOT / e["path"]
            W, H = image_size(path)
            boxes, labels = [], []
            for cid, cx, cy, w, h in entry_boxes(cfg, e):
                c = clip_xywh((cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H, W, H)
                if c:
                    boxes.append(c)
                    labels.append(cid)
            self.items.append((path, W, H, boxes, labels))
        bp = A.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.3, min_area=4.0)
        if train:
            self.tf = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
                A.HueSaturationValue(10, 30, 20, p=0.5),
                A.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), p=0.5),
                A.Resize(imgsz, imgsz),
            ], bbox_params=bp)
        else:
            self.tf = A.Compose([A.Resize(imgsz, imgsz)], bbox_params=bp)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        import cv2
        import torch

        path, W, H, boxes, labels = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        res = self.tf(image=img, bboxes=boxes, labels=labels)
        pv = torch.from_numpy(res["image"]).permute(2, 0, 1).float() / 255.0  # processor: rescale only
        bx = to_normalized_cxcywh(np.array(res["bboxes"]) if len(res["bboxes"]) else np.zeros((0, 4)),
                                  self.imgsz, self.imgsz)
        target = {"class_labels": torch.tensor(res["labels"], dtype=torch.long),
                  "boxes": torch.from_numpy(bx)}
        return pv, target, {"index": i, "size": (W, H)}


def collate(batch):
    import torch
    return (torch.stack([b[0] for b in batch]), [b[1] for b in batch], [b[2] for b in batch])


@_no_grad
def validate(model, loader, gt, device) -> dict[str, float]:
    """COCO AP on the val split (fitness for early stopping / best checkpoint)."""
    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    results = []
    for pv, _, metas in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(pixel_values=pv.to(device))
        dec = decode_predictions(out.logits, out.pred_boxes, [m["size"] for m in metas])
        for rows, m in zip(dec, metas):
            for c, s, x, y, w, h in rows:
                if s < 0.001:
                    continue
                results.append({"image_id": m["index"], "category_id": int(c) + CAT_OFFSET,
                                "bbox": [float(x), float(y), float(w), float(h)], "score": float(s)})
    if not results:
        return {"ap": 0.0, "ap50": 0.0}
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    ev = COCOeval(coco_gt, coco_gt.loadRes(results), "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"ap": float(ev.stats[0]), "ap50": float(ev.stats[1])}


def train(cfg: Config, run_id: str, run_dir: Path, manifest: dict, mb: int, accum: int,
          rec: dict[str, Any]) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import DFineForObjectDetection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgsz = int(cfg["imgsz"])
    b = cfg["budget"]
    epochs, patience = int(b["max_epochs"]), int(b["patience"])
    seed = rec["seed"]

    id2label = dict(enumerate(cfg.classes))
    model = DFineForObjectDetection.from_pretrained(
        str(pretrained_path(cfg, run_id)), num_labels=len(cfg.classes),
        id2label=id2label, label2id={v: k for k, v in id2label.items()},
        ignore_mismatched_sizes=True,
    ).to(device)

    train_ds = DetDataset(cfg, manifest, "train", imgsz, train=True)
    val_ds = DetDataset(cfg, manifest, "val", imgsz, train=False)
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=mb, shuffle=True, num_workers=4, collate_fn=collate,
                          generator=g, drop_last=len(train_ds) >= mb, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=mb, shuffle=False, num_workers=2, collate_fn=collate)
    gt_val = gt_coco(cfg, manifest, "val")

    backbone = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": backbone, "lr": LR * LR_BACKBONE_MULT}, {"params": rest, "lr": LR}],
        weight_decay=WEIGHT_DECAY)
    base_lrs = [g_["lr"] for g_ in opt.param_groups]
    steps_per_epoch = max(1, len(train_dl) // accum)
    total_steps = steps_per_epoch * epochs

    wb = start_wandb(cfg, run_id, rec, {"lr": LR, "lr_backbone_mult": LR_BACKBONE_MULT,
                                        "weight_decay": WEIGHT_DECAY, "optimizer": "AdamW"})

    best_dir = run_dir / "dfine" / "best"
    best_ap, bad, step = -1.0, 0, 0
    for epoch in range(epochs):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for it, (pv, labels, _) in enumerate(train_dl):
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(pixel_values=pv.to(device), labels=labels)
            (out.loss / accum).backward()
            run_loss += float(out.loss.detach()); n += 1
            if (it + 1) % accum == 0:
                f = lr_factor(step, total_steps)
                for g_, base in zip(opt.param_groups, base_lrs):
                    g_["lr"] = base * f
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
        m = validate(model, val_dl, gt_val, device)
        msg = {"epoch": epoch, "train_loss": run_loss / max(1, n), "val_ap": m["ap"],
               "val_ap50": m["ap50"], "epoch_sec": time.time() - t0}
        print(msg, flush=True)
        if wb:
            wb.log(msg)
        if m["ap"] > best_ap:
            best_ap, bad = m["ap"], 0
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch} (best val AP {best_ap:.4f})", flush=True)
                break
    wid = wb.id if wb else None
    if wb:
        wb.finish()
    if not (best_dir / "model.safetensors").exists():
        raise RuntimeError("no checkpoint was saved (validation AP never computed)")
    return {"weights_path": best_dir, "wandb_run_id": wid, "input_size": imgsz,
            "notes": f"D-FINE HF fine-tune, best val AP@[.5:.95]={best_ap:.4f}; augmentation: "
                     "hflip, brightness/contrast, HSV, affine scale/translate (albumentations)"}
