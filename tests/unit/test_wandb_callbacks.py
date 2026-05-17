"""Unit tests for `core.train.callbacks`.

Covers:
- val image logger fires only at the configured epoch multiples
- val image logger is a no-op when there are no images or no active wandb.run
- setup_wandb_mode honours `disabled=True`
- setup_wandb_mode falls back to offline when `wandb.login` raises
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from core.config import WandbConfig
from core.train.callbacks import make_val_image_logger, setup_wandb_mode


@dataclass
class _FakeTrainer:
    epoch: int
    save_dir: Path


class _FakeImage:
    def __init__(self, path: str) -> None:
        self.path = path


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, Any], int | None]] = []

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        self.logged.append((payload, step))


def _install_fake_wandb(monkeypatch: pytest.MonkeyPatch, *, run: _FakeRun | None, login_raises: bool = False) -> types.ModuleType:
    fake = types.ModuleType("wandb")

    def _login(timeout: int = 5) -> bool:
        if login_raises:
            raise RuntimeError("simulated wandb unreachable")
        return True

    fake.login = _login  # type: ignore[attr-defined]
    fake.Image = _FakeImage  # type: ignore[attr-defined]
    fake.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def _make_val_imgs(save_dir: Path, n: int) -> list[Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = save_dir / f"val_batch{i}_pred.jpg"
        p.write_bytes(b"\xff\xd8\xff\xd9")
        paths.append(p)
    return paths


def test_val_image_logger_fires_at_multiples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _FakeRun()
    _install_fake_wandb(monkeypatch, run=run)
    _make_val_imgs(tmp_path, 3)

    cb = make_val_image_logger(log_every_n=10)
    # epochs 0..29 -> the trigger condition is (epoch + 1) % 10 == 0,
    # so epoch=9, 19, 29 fire.
    for epoch in range(30):
        cb(_FakeTrainer(epoch=epoch, save_dir=tmp_path))

    fired_epochs = [step for _, step in run.logged]
    assert fired_epochs == [10, 20, 30]
    # Each call logs 3 images.
    for payload, _ in run.logged:
        assert len(payload["val_predictions"]) == 3


def test_val_image_logger_skips_non_multiples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _FakeRun()
    _install_fake_wandb(monkeypatch, run=run)
    _make_val_imgs(tmp_path, 1)

    cb = make_val_image_logger(log_every_n=10)
    for epoch in [0, 1, 5, 8]:
        cb(_FakeTrainer(epoch=epoch, save_dir=tmp_path))

    assert run.logged == []


def test_val_image_logger_noop_when_no_active_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_wandb(monkeypatch, run=None)
    _make_val_imgs(tmp_path, 2)

    cb = make_val_image_logger(log_every_n=5)
    # Should not raise even though wandb.run is None.
    cb(_FakeTrainer(epoch=4, save_dir=tmp_path))


def test_val_image_logger_noop_when_no_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _FakeRun()
    _install_fake_wandb(monkeypatch, run=run)
    # tmp_path exists but no val_batch* images present.

    cb = make_val_image_logger(log_every_n=5)
    cb(_FakeTrainer(epoch=4, save_dir=tmp_path))
    assert run.logged == []


def test_setup_wandb_mode_disabled_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)
    cfg = WandbConfig(project="p", entity="", log_val_images_every_n=10)
    mode = setup_wandb_mode(cfg, disabled=True)
    assert mode == "disabled"
    import os

    assert os.environ.get("WANDB_MODE") == "disabled"


def test_setup_wandb_mode_offline_on_login_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)
    _install_fake_wandb(monkeypatch, run=None, login_raises=True)

    cfg = WandbConfig(project="p", entity="my-team", log_val_images_every_n=10)
    mode = setup_wandb_mode(cfg, disabled=False)
    assert mode == "offline"
    import os

    assert os.environ.get("WANDB_MODE") == "offline"
    # Entity should have been propagated to env from config.
    assert os.environ.get("WANDB_ENTITY") == "my-team"


def test_setup_wandb_mode_online_when_login_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)
    _install_fake_wandb(monkeypatch, run=None, login_raises=False)

    cfg = WandbConfig(project="p", entity="", log_val_images_every_n=10)
    mode = setup_wandb_mode(cfg, disabled=False)
    assert mode == "online"
