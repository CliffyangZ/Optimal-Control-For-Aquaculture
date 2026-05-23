"""Evaluation and quality gate for the YOLO26 training pipeline.

After training, `Evaluator` loads `best.pt`, runs `model.val()` on the validation
split, and checks each class against the quality thresholds in `config/training.yaml`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.config import EvaluationConfig

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


@dataclass
class ClassMetrics:
    class_name: str
    precision: float
    recall: float
    f1: float
    map50: float


@dataclass
class MetricsReport:
    classes: dict[str, ClassMetrics]
    quality_gate_passed: bool
    threshold_violations: list[str] = field(default_factory=list)


CLASS_NAMES = ("fish", "feed", "debris", "lumpfish")


class Evaluator:
    """Evaluate a trained YOLO26 model and apply the quality gate."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def evaluate(self, model_path: Path, val_yaml: Path) -> MetricsReport:
        """Load `model_path`, run `model.val(data=val_yaml)`, return `MetricsReport`."""
        from ultralytics import YOLO  # ml group only

        model = YOLO(str(model_path))
        results = model.val(data=str(val_yaml))

        # Ultralytics stores per-class metrics on results.box
        box = results.box
        names: dict[int, str] = results.names  # {index: class_name}

        # Ultralytics returns arrays indexed by class position
        p_arr = np.asarray(box.p) if hasattr(box, "p") else np.array([])
        r_arr = np.asarray(box.r) if hasattr(box, "r") else np.array([])
        f1_arr = np.asarray(box.f1) if hasattr(box, "f1") else np.array([])
        ap50_arr = np.asarray(box.ap50) if hasattr(box, "ap50") else np.array([])

        classes: dict[str, ClassMetrics] = {}
        for idx, name in names.items():
            if name not in CLASS_NAMES:
                continue
            classes[name] = ClassMetrics(
                class_name=name,
                precision=float(p_arr[idx]) if idx < len(p_arr) else 0.0,
                recall=float(r_arr[idx]) if idx < len(r_arr) else 0.0,
                f1=float(f1_arr[idx]) if idx < len(f1_arr) else 0.0,
                map50=float(ap50_arr[idx]) if idx < len(ap50_arr) else 0.0,
            )

        # Fill in any missing classes with zeros
        for name in CLASS_NAMES:
            if name not in classes:
                classes[name] = ClassMetrics(class_name=name, precision=0.0, recall=0.0, f1=0.0, map50=0.0)

        violations: list[str] = []
        for name in CLASS_NAMES:
            cm = classes[name]
            threshold_cfg = self.config.thresholds.get(name)
            if not threshold_cfg:
                continue

            if threshold_cfg.map50 is not None and cm.map50 < threshold_cfg.map50:
                violations.append(
                    f"{name}: map50={cm.map50:.3f} < threshold {threshold_cfg.map50}"
                )
            if threshold_cfg.recall is not None and cm.recall < threshold_cfg.recall:
                violations.append(
                    f"{name}: recall={cm.recall:.3f} < threshold {threshold_cfg.recall}"
                )

        return MetricsReport(
            classes=classes,
            quality_gate_passed=len(violations) == 0,
            threshold_violations=violations,
        )

    def write_report(self, report: MetricsReport, path: Path) -> None:
        """Serialize `MetricsReport` to JSON and print the table to stdout."""
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "quality_gate_passed": report.quality_gate_passed,
            "threshold_violations": report.threshold_violations,
            "classes": {
                name: {
                    "precision": cm.precision,
                    "recall": cm.recall,
                    "f1": cm.f1,
                    "map50": cm.map50,
                }
                for name, cm in report.classes.items()
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        self.print_table(report)
        print(f"[evaluate] Report saved: {path}")

    def print_table(self, report: MetricsReport) -> None:
        """Print a formatted metrics table to stdout."""
        print("[evaluate] Evaluation metrics:")
        print()

        header = (
            f"  {'Class':<12} {'mAP@0.5':<14} "
            f"{'Recall':<14} {'Threshold':<14} {'Result'}"
        )
        rule = "  " + "─" * (len(header) - 2)
        print(header)
        print(rule)

        for name in CLASS_NAMES:
            cm = report.classes[name]
            map50_str = f"{cm.map50:.3f}"
            rec_str = f"{cm.recall:.3f}"

            threshold_cfg = self.config.thresholds.get(name)
            parts = []
            if threshold_cfg and threshold_cfg.map50 is not None:
                parts.append(f"≥ {threshold_cfg.map50}")
            if threshold_cfg and threshold_cfg.recall is not None:
                parts.append(f"recall ≥ {threshold_cfg.recall}")
            thresh_str = " / ".join(parts) if parts else "—"

            # Determine if this class passes threshold
            passes = True
            if threshold_cfg:
                if threshold_cfg.map50 is not None and cm.map50 < threshold_cfg.map50:
                    passes = False
                if threshold_cfg.recall is not None and cm.recall < threshold_cfg.recall:
                    passes = False
            result_str = "PASS ✓" if passes else "FAIL ✗"

            print(
                f"  {name:<12} {map50_str:<14} {rec_str:<14} {thresh_str:<14} {result_str}"
            )

        print()
        gate = "PASSED" if report.quality_gate_passed else "FAILED"
        print(f"[evaluate] Quality gate: {gate}")
        if report.threshold_violations:
            for v in report.threshold_violations:
                print(f"[evaluate]   ✗ {v}")
