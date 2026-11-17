#!/usr/bin/env python3
"""Download YOLO26 / RF-DETR / D-FINE n·s·m pretrained weights into models/pretrained."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "pretrained"

RFDETR_URLS = {
    "rf-detr-nano.pth": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
    "rf-detr-small.pth": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
    "rf-detr-medium.pth": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
}

DFINE_REPOS = {
    "dfine-n": "ustc-community/dfine-nano-coco",
    "dfine-s": "ustc-community/dfine-small-coco",
    "dfine-m": "ustc-community/dfine-medium-coco",
}

YOLO26 = ("yolo26n.pt", "yolo26s.pt", "yolo26m.pt")


def _download_url(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  ↓ {url}")
    print(f"    → {dest}")
    with urllib.request.urlopen(url, timeout=600) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)


def download_yolo26() -> None:
    from ultralytics import YOLO

    dest_dir = OUT / "yolo26"
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("\n[YOLO26]")
    for name in YOLO26:
        dest = dest_dir / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"  ✓ exists {dest}")
            continue
        # Ultralytics caches to CWD / weights name; load then copy.
        print(f"  ↓ ultralytics {name}")
        model = YOLO(name)
        src = Path(name)
        if not src.exists():
            # resolve from model.ckpt path if available
            ckpt = getattr(model, "ckpt_path", None) or getattr(getattr(model, "model", None), "pt_path", None)
            if ckpt and Path(ckpt).exists():
                src = Path(ckpt)
        if not src.exists():
            raise FileNotFoundError(f"Could not locate downloaded {name}")
        shutil.copy2(src, dest)
        print(f"  ✓ {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def download_rfdetr() -> None:
    dest_dir = OUT / "rfdetr"
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("\n[RF-DETR]")
    for name, url in RFDETR_URLS.items():
        dest = dest_dir / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"  ✓ exists {dest}")
            continue
        _download_url(url, dest)
        print(f"  ✓ {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def download_dfine() -> None:
    from huggingface_hub import snapshot_download

    print("\n[D-FINE]")
    for folder, repo in DFINE_REPOS.items():
        dest = OUT / "dfine" / folder
        if (dest / "model.safetensors").exists() or (dest / "pytorch_model.bin").exists():
            print(f"  ✓ exists {dest}")
            continue
        print(f"  ↓ {repo} → {dest}")
        snapshot_download(
            repo_id=repo,
            local_dir=str(dest),
            allow_patterns=["*.safetensors", "*.bin", "config.json", "preprocessor_config.json", "README.md"],
        )
        print(f"  ✓ {dest}")


def write_readme() -> None:
    readme = OUT / "README.md"
    readme.write_text(
        """# Pretrained Weights

COCO（或官方）預訓練權重，對應 `Training_Method.md` 的 n／s／m 比較矩陣。

| 族系 | 路徑 | 來源 |
| --- | --- | --- |
| YOLO26 | `yolo26/yolo26{n,s,m}.pt` | Ultralytics |
| RF-DETR | `rfdetr/rf-detr-{nano,small,medium}.pth` | Roboflow GCS |
| D-FINE | `dfine/dfine-{n,s,m}/` | Hugging Face `ustc-community/dfine-*-coco` |

重新下載：

```bash
.venv/bin/python scripts/download_pretrained.py
```
""",
        encoding="utf-8",
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    download_yolo26()
    download_rfdetr()
    download_dfine()
    write_readme()
    print("\nDone. Layout:")
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "README.md":
            print(f"  {p.relative_to(OUT)}  ({p.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
