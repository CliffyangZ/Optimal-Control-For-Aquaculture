"""FishTracker_v2 dataset: Roboflow download, validation, and 5-fold CV split generation.

The dataset is pinned to a specific Roboflow version. After download, every image
must have a matching YOLO `.txt` annotation. K-fold splits are written under
`data/kfold/fold_{i}/` as symlinks (or copies on platforms without symlink support)
with companion `fold_{i}.yaml` files consumable directly by Ultralytics' `model.train()`.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from core.config import TrainingConfig
from core.types import CLASS_NAMES


IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class DatasetValidationSummary:
    image_count: int
    annotation_count: int
    classes: list[str]
    dataset_yaml: Path


class FishTrackerDataset:
    """Roboflow-backed FishTracker_v2 dataset with k-fold split generation."""

    def __init__(self, config: TrainingConfig, project_root: Path | None = None):
        self.config = config
        self.project_root = (project_root or Path.cwd()).resolve()
        self.local_cache_dir = (self.project_root / config.dataset.local_cache_dir).resolve()
        self.fold_output_dir = (self.project_root / config.dataset.fold_output_dir).resolve()

    # ------------------------------------------------------------------ download
    def download(self) -> Path:
        """Download the pinned FishTracker_v2 version from Roboflow.

        Returns the local cache directory. Requires `ROBOFLOW_API_KEY` in env.
        Raises RuntimeError if the env var is missing.
        """
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ROBOFLOW_API_KEY is not set. Add it to .env or export it before running download."
            )

        # Import lazily so the rest of the module is importable without the ml group installed.
        from roboflow import Roboflow

        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        rf = Roboflow(api_key=api_key)
        project = rf.workspace(self.config.dataset.roboflow_workspace).project(
            self.config.dataset.roboflow_project
        )
        version = project.version(self.config.dataset.roboflow_version)
        # Roboflow downloads into a directory named like "fishtracker-v2-1"; we
        # let it place the export inside our cache dir.
        version.download(
            self.config.dataset.roboflow_format,
            location=str(self.local_cache_dir),
            overwrite=False,
        )
        return self.local_cache_dir

    # ---------------------------------------------------------------- validation
    def _dataset_yaml_path(self) -> Path:
        candidates = [
            self.local_cache_dir / "data.yaml",
            self.local_cache_dir / "dataset.yaml",
        ]
        for c in candidates:
            if c.exists():
                return c
        # Roboflow puts data.yaml at the export root, which sometimes nests once.
        for p in self.local_cache_dir.rglob("data.yaml"):
            return p
        raise FileNotFoundError(
            f"No data.yaml/dataset.yaml found under {self.local_cache_dir}"
        )

    def _all_image_paths(self) -> list[Path]:
        images: list[Path] = []
        for ext in IMAGE_EXTS:
            images.extend(self.local_cache_dir.rglob(f"*{ext}"))
        return sorted(images)

    @staticmethod
    def _label_path_for(image_path: Path) -> Path:
        # Roboflow YOLOv8 export layout: .../split/images/foo.jpg ↔ .../split/labels/foo.txt
        parts = list(image_path.parts)
        try:
            i = len(parts) - 1 - parts[::-1].index("images")
            parts[i] = "labels"
        except ValueError:
            return image_path.with_suffix(".txt")
        return Path(*parts).with_suffix(".txt")

    def validate(self) -> DatasetValidationSummary:
        """Check classes and that every image has a matching annotation file.

        Raises ValueError if a class is missing or any image lacks its `.txt`.
        """
        dataset_yaml = self._dataset_yaml_path()
        with open(dataset_yaml, "r", encoding="utf-8") as f:
            ds_meta = yaml.safe_load(f)

        names = ds_meta.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        missing_classes = [c for c in CLASS_NAMES if c not in names]
        if missing_classes:
            raise ValueError(
                f"Dataset missing required classes {missing_classes}; got names={names}"
            )

        images = self._all_image_paths()
        if not images:
            raise ValueError(f"No images found under {self.local_cache_dir}")

        missing_labels = [
            img for img in images if not self._label_path_for(img).exists()
        ]
        if missing_labels:
            raise ValueError(
                f"{len(missing_labels)} image(s) missing label files; "
                f"first missing: {missing_labels[0]}"
            )

        return DatasetValidationSummary(
            image_count=len(images),
            annotation_count=len(images),
            classes=list(names),
            dataset_yaml=dataset_yaml,
        )

    # ----------------------------------------------------------------- k-fold CV
    def generate_kfold_splits(self) -> list[Path]:
        """Generate 5-fold CV splits.

        Writes one `data/kfold/fold_{i}.yaml` per fold and links each fold's
        train/val image files under `data/kfold/fold_{i}/{train,val}/images` along
        with matching labels under `.../labels`.

        Returns the list of fold YAML paths.
        """
        from sklearn.model_selection import KFold  # local import: ml group only

        images = self._all_image_paths()
        if not images:
            raise ValueError(f"No images found under {self.local_cache_dir}")

        n_folds = self.config.dataset.n_folds
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.config.model.seed)

        self.fold_output_dir.mkdir(parents=True, exist_ok=True)
        fold_yaml_paths: list[Path] = []
        seen_val_stems: set[str] = set()

        for k, (train_idx, val_idx) in enumerate(kf.split(images)):
            fold_dir = self.fold_output_dir / f"fold_{k}"
            for split, indices in (("train", train_idx), ("val", val_idx)):
                img_dst = fold_dir / split / "images"
                lbl_dst = fold_dir / split / "labels"
                self._reset_dir(img_dst)
                self._reset_dir(lbl_dst)
                for idx in indices:
                    img = images[idx]
                    lbl = self._label_path_for(img)
                    self._link_or_copy(img, img_dst / img.name)
                    if lbl.exists():
                        self._link_or_copy(lbl, lbl_dst / lbl.name)
                    if split == "val":
                        seen_val_stems.add(img.stem)

            yaml_path = self.fold_output_dir / f"fold_{k}.yaml"
            yaml_path.write_text(
                yaml.safe_dump(
                    {
                        "path": str(self.project_root),
                        "train": str(
                            (fold_dir / "train" / "images").relative_to(self.project_root)
                        ),
                        "val": str(
                            (fold_dir / "val" / "images").relative_to(self.project_root)
                        ),
                        "nc": len(CLASS_NAMES),
                        "names": list(CLASS_NAMES),
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            fold_yaml_paths.append(yaml_path)

        # Invariant: every image stem appears in exactly one fold's val set.
        if len(seen_val_stems) != len(images):
            raise RuntimeError(
                f"KFold invariant broken: {len(seen_val_stems)} unique val stems "
                f"vs {len(images)} images"
            )

        return fold_yaml_paths

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _reset_dir(p: Path) -> None:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        try:
            os.symlink(src.resolve(), dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)
