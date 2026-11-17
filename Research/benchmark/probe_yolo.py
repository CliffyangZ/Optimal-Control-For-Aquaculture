"""Probe: does more epochs / higher imgsz fix under-training? Not part of the benchmark runs.
usage: python probe_yolo.py <name> <epochs> <imgsz>   (Y-s settings otherwise; no W&B)"""
import json, os, sys
os.environ.setdefault("MPLBACKEND", "Agg")  # headless: Ultralytics plotting otherwise tries tkagg
from pathlib import Path
from benchmark.config import REPO_ROOT, load_config
from benchmark.data.split import load_manifest
from benchmark.data.export import export_yolo
from benchmark.train.yolo import build_train_args
from benchmark.infer.common import eval_set_images, eval_set_gt
from benchmark.infer.adapter_yolo import predict
from benchmark.eval.coco_ap import coco_ap
from benchmark.eval.prf_ci import prf_with_ci

def main():
    name, epochs, imgsz = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    eval_only = len(sys.argv) > 4 and sys.argv[4] == "eval"
    micro = int(sys.argv[5]) if len(sys.argv) > 5 else (16 if imgsz <= 640 else 4)
    cfg = load_config()
    out = cfg.results_root / "probe" / name
    args = build_train_args(cfg, "Y-s", export_yolo(cfg, load_manifest(cfg)), out, 16, 42)
    args.update(epochs=epochs, imgsz=imgsz, patience=epochs, name="train", batch=micro)
    from ultralytics import YOLO, settings
    settings.update({"wandb": False, "mlflow": False})
    if not eval_only:
        YOLO(str(REPO_ROOT / cfg.run_spec("Y-s")["pretrained"])).train(**args)
    rec = {"weights_path": str(out / "train" / "weights" / "best.pt"), "input_size": imgsz}
    gt = eval_set_gt(cfg, "heldout_real")
    preds = predict(cfg, rec, eval_set_images(cfg, "heldout_real"))
    ap = coco_ap(gt, preds, cfg.classes)
    prf = prf_with_ci(gt, preds, cfg.classes, 0.25, 0.5, 200, 123, ["feed", "fish"])
    (out / "heldout.json").write_text(json.dumps({"epochs": epochs, "imgsz": imgsz, "ap": ap, "prf": prf}, indent=1))
    print("DONE", name, ap["overall"]["ap_50_95"], ap["feed"]["ap_50_95"], prf["feed"]["f1"]["value"])


if __name__ == "__main__":
    main()
