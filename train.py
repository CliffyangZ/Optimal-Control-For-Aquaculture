"""CLI entry point for the YOLO Fish & Feed Detection pipeline.

Subcommands:
    download   Download FishTracker_v2 from Roboflow and generate 5-fold splits
    train      Train YOLO26s across folds with wandb logging
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.config import load_training_config


def _cmd_download(args: argparse.Namespace) -> int:
    from core.train.dataset import FishTrackerDataset

    config = load_training_config(args.config)
    dataset = FishTrackerDataset(config)

    try:
        cache_dir = dataset.download()
        summary = dataset.validate()
        fold_yamls = dataset.generate_kfold_splits()
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(f"[download] ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"[download] Dataset cached: {cache_dir} "
        f"({summary.image_count} images, {len(summary.classes)} classes)"
    )
    print(
        f"[download] 5-fold splits generated: {fold_yamls[0].parent} "
        f"({len(fold_yamls)} fold YAML configs)"
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from core.train.callbacks import setup_wandb_mode
    from core.train.trainer import FoldTrainer

    config = load_training_config(args.config)
    setup_wandb_mode(config.wandb, disabled=args.no_wandb)

    trainer = FoldTrainer(config, project_root=Path.cwd())
    try:
        if args.fold is not None:
            fold_yaml = Path(config.dataset.fold_output_dir) / f"fold_{args.fold}.yaml"
            if not fold_yaml.exists():
                print(
                    f"[train] ERROR: fold YAML not found: {fold_yaml}. "
                    f"Run `python train.py download` first.",
                    file=sys.stderr,
                )
                return 1
            trainer.train_fold(args.fold, fold_yaml)
        else:
            trainer.train_all_folds()
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(f"[train] ERROR: {exc}", file=sys.stderr)
        return 2

    print("[train] All folds complete. Run `python train.py evaluate` for cross-fold report.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train.py", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download dataset + generate 5-fold splits")
    p_dl.add_argument("--config", type=Path, default=Path("config/training.yaml"))
    p_dl.set_defaults(func=_cmd_download)

    p_tr = sub.add_parser("train", help="Train YOLO26s across folds with wandb")
    p_tr.add_argument("--config", type=Path, default=Path("config/training.yaml"))
    p_tr.add_argument(
        "--fold",
        type=int,
        choices=range(0, 5),
        default=None,
        help="Run a single fold (0–4); omit to run all 5",
    )
    p_tr.add_argument(
        "--no-wandb", action="store_true", help="Disable wandb logging entirely"
    )
    p_tr.set_defaults(func=_cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
