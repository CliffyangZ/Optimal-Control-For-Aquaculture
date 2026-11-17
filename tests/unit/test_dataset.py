"""Unit tests for FishTrackerDataset.

Targets the missing-annotation guard and dataset validation.
Roboflow API calls are mocked; no network or GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


from core.config import (
    DatasetConfig,
    EvaluationConfig,
    ModelConfig,
    TrainingConfig,
    WandbConfig,
)
from core.train.dataset import FishTrackerDataset
from core.types import CLASS_NAMES


N_IMAGES = 25


def _make_training_config(local_cache: Path) -> TrainingConfig:
    return TrainingConfig(
        dataset=DatasetConfig(
            roboflow_workspace="jtelectric",
            roboflow_project="fishtracker-v2",
            roboflow_version=1,
            roboflow_format="yolov8",
            local_cache_dir=local_cache,
        ),
        model=ModelConfig(
            weights="yolo26s.pt",
            variant="yolo26s",
            output_dir="Results/runs/train",
            epochs=1,
            batch=2,
            imgsz=320,
            workers=0,
            mosaic=0.0,
            close_mosaic=0,
            mixup=0.0,
            patience=1,
            seed=42,
        ),
        wandb=WandbConfig(project="aquaculture-cv", entity="", log_val_images_every_n=10),
        evaluation=EvaluationConfig(thresholds={}),
    )


def _make_fake_dataset(root: Path, *, missing_label_for: str | None = None) -> Path:
    """Create a Roboflow-shaped yolov8 export with N_IMAGES dummy frames."""
    train_imgs = root / "train" / "images"
    train_lbls = root / "train" / "labels"
    train_imgs.mkdir(parents=True)
    train_lbls.mkdir(parents=True)

    for i in range(N_IMAGES):
        stem = f"img_{i:03d}"
        (train_imgs / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # tiny JPEG
        if stem != missing_label_for:
            (train_lbls / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "train/images",
                "nc": len(CLASS_NAMES),
                "names": list(CLASS_NAMES),
            }
        )
    )
    return root


def test_validate_raises_on_missing_annotation(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _make_fake_dataset(cache, missing_label_for="img_005")
    cfg = _make_training_config(cache)
    ds = FishTrackerDataset(cfg, project_root=tmp_path)

    with pytest.raises(ValueError, match="missing label files"):
        ds.validate()


def test_validate_succeeds_when_all_labels_present(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _make_fake_dataset(cache)
    cfg = _make_training_config(cache)
    ds = FishTrackerDataset(cfg, project_root=tmp_path)

    summary = ds.validate()
    assert summary.image_count == N_IMAGES
    assert summary.classes == list(CLASS_NAMES)


def test_download_uses_mocked_roboflow_when_api_key_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Download() must read ROBOFLOW_API_KEY and dispatch to Roboflow client."""
    cache = tmp_path / "cache"
    cfg = _make_training_config(cache)
    ds = FishTrackerDataset(cfg, project_root=tmp_path)

    calls: dict[str, object] = {}

    class _FakeVersion:
        def download(self, fmt: str, location: str, overwrite: bool) -> None:  # noqa: D401
            calls["download"] = (fmt, location, overwrite)
            _make_fake_dataset(Path(location))

    class _FakeProject:
        def version(self, v: int) -> _FakeVersion:
            calls["version"] = v
            return _FakeVersion()

    class _FakeWorkspace:
        def project(self, p: str) -> _FakeProject:
            calls["project"] = p
            return _FakeProject()

    class _FakeRoboflow:
        def __init__(self, api_key: str) -> None:
            calls["api_key"] = api_key

        def workspace(self, ws: str) -> _FakeWorkspace:
            calls["workspace"] = ws
            return _FakeWorkspace()

    fake_module = type(sys)("roboflow")
    fake_module.Roboflow = _FakeRoboflow
    monkeypatch.setitem(sys.modules, "roboflow", fake_module)
    monkeypatch.setenv("ROBOFLOW_API_KEY", "fake-key-123")

    out = ds.download()
    assert out == cache.resolve()
    assert calls["api_key"] == "fake-key-123"
    assert calls["workspace"] == "jtelectric"
    assert calls["project"] == "fishtracker-v2"
    assert calls["version"] == 1
    assert calls["download"][0] == "yolov8"


def test_download_raises_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_training_config(tmp_path / "cache")
    ds = FishTrackerDataset(cfg, project_root=tmp_path)

    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ROBOFLOW_API_KEY"):
        ds.download()
