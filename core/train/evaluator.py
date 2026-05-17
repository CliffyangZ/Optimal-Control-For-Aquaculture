"""Cross-fold evaluation and quality gate for the YOLO26 training pipeline.

After all 5 folds are trained, `CrossFoldEvaluator` loads each fold's `best.pt`,
runs `model.val()` on its validation split, aggregates per-class metrics across
folds, and checks each class against the quality thresholds in `config/training.yaml`.
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
class FoldMetrics:
    fold_index: int
    classes: dict[str, ClassMetrics]
    best_epoch: int = 0


@dataclass
class AggregateClassMetrics:
    class_name: str
    mean_map50: float
    std_map50: float
    mean_recall: float
    std_recall: float
    mean_precision: float
    mean_f1: float
    threshold_map50: float | None = None
    threshold_recall: float | None = None
    passes_threshold: bool | None = None


@dataclass
class MetricsReport:
    fold_metrics: list[FoldMetrics]
    aggregate: dict[str, AggregateClassMetrics]
    quality_gate_passed: bool
    threshold_violations: list[str] = field(default_factory=list)


CLASS_NAMES = ("fish", "feed", "debris", "lumpfish")


class CrossFoldEvaluator:
    """Aggregate metrics across trained fold models and apply the quality gate."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def evaluate_fold(self, k: int, model_path: Path, val_yaml: Path) -> FoldMetrics:
        """Load `model_path`, run `model.val(data=val_yaml)`, return `FoldMetrics`."""
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

        return FoldMetrics(fold_index=k, classes=classes)

    def aggregate(self, fold_metrics_list: list[FoldMetrics]) -> MetricsReport:
        """Compute mean ± std per class per metric; apply quality gate thresholds."""
        per_class: dict[str, dict[str, list[float]]] = {
            name: {"precision": [], "recall": [], "f1": [], "map50": []}
            for name in CLASS_NAMES
        }

        for fm in fold_metrics_list:
            for name in CLASS_NAMES:
                cm = fm.classes.get(name)
                if cm is None:
                    for metric in ("precision", "recall", "f1", "map50"):
                        per_class[name][metric].append(0.0)
                else:
                    per_class[name]["precision"].append(cm.precision)
                    per_class[name]["recall"].append(cm.recall)
                    per_class[name]["f1"].append(cm.f1)
                    per_class[name]["map50"].append(cm.map50)

        aggregate: dict[str, AggregateClassMetrics] = {}
        violations: list[str] = []

        for name in CLASS_NAMES:
            vals = per_class[name]
            m50 = np.array(vals["map50"])
            rec = np.array(vals["recall"])
            prec = np.array(vals["precision"])
            f1 = np.array(vals["f1"])

            threshold_cfg = self.config.thresholds.get(name)
            thresh_map50 = threshold_cfg.map50 if threshold_cfg else None
            thresh_recall = threshold_cfg.recall if threshold_cfg else None

            mean_map50 = float(m50.mean())
            mean_recall = float(rec.mean())

            passes: bool | None = None
            if thresh_map50 is not None or thresh_recall is not None:
                passes = True
                if thresh_map50 is not None and mean_map50 < thresh_map50:
                    violations.append(
                        f"{name}: mean_map50={mean_map50:.3f} < threshold {thresh_map50}"
                    )
                    passes = False
                if thresh_recall is not None and mean_recall < thresh_recall:
                    violations.append(
                        f"{name}: mean_recall={mean_recall:.3f} < threshold {thresh_recall}"
                    )
                    passes = False

            aggregate[name] = AggregateClassMetrics(
                class_name=name,
                mean_map50=mean_map50,
                std_map50=float(m50.std()),
                mean_recall=mean_recall,
                std_recall=float(rec.std()),
                mean_precision=float(prec.mean()),
                mean_f1=float(f1.mean()),
                threshold_map50=thresh_map50,
                threshold_recall=thresh_recall,
                passes_threshold=passes,
            )

        return MetricsReport(
            fold_metrics=fold_metrics_list,
            aggregate=aggregate,
            quality_gate_passed=len(violations) == 0,
            threshold_violations=violations,
        )

    def write_report(self, report: MetricsReport, path: Path) -> None:
        """Serialize `MetricsReport` to JSON and print the table to stdout."""
        path.parent.mkdir(parents=True, exist_ok=True)

        def _fm_to_dict(fm: FoldMetrics) -> dict:
            return {
                "fold_index": fm.fold_index,
                "best_epoch": fm.best_epoch,
                "classes": {
                    name: {
                        "precision": cm.precision,
                        "recall": cm.recall,
                        "f1": cm.f1,
                        "map50": cm.map50,
                    }
                    for name, cm in fm.classes.items()
                },
            }

        def _agg_to_dict(a: AggregateClassMetrics) -> dict:
            return {
                "class_name": a.class_name,
                "mean_map50": a.mean_map50,
                "std_map50": a.std_map50,
                "mean_recall": a.mean_recall,
                "std_recall": a.std_recall,
                "mean_precision": a.mean_precision,
                "mean_f1": a.mean_f1,
                "threshold_map50": a.threshold_map50,
                "threshold_recall": a.threshold_recall,
                "passes_threshold": a.passes_threshold,
            }

        payload = {
            "quality_gate_passed": report.quality_gate_passed,
            "threshold_violations": report.threshold_violations,
            "aggregate": {name: _agg_to_dict(a) for name, a in report.aggregate.items()},
            "fold_metrics": [_fm_to_dict(fm) for fm in report.fold_metrics],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        self.print_table(report)
        print(f"[evaluate] Report saved: {path}")

    def print_table(self, report: MetricsReport) -> None:
        """Print a formatted cross-validation metrics table to stdout."""
        print("[evaluate] Cross-validation metrics (5 folds):")
        print()

        header = (
            f"  {'Class':<12} {'mAP@0.5 (mean ± std)':<26} "
            f"{'Recall (mean ± std)':<24} {'Threshold':<14} {'Result'}"
        )
        rule = "  " + "─" * (len(header) - 2)
        print(header)
        print(rule)

        for name in CLASS_NAMES:
            agg = report.aggregate[name]
            map50_str = f"{agg.mean_map50:.3f} ± {agg.std_map50:.3f}"
            rec_str = f"{agg.mean_recall:.3f} ± {agg.std_recall:.3f}"

            parts = []
            if agg.threshold_map50 is not None:
                parts.append(f"≥ {agg.threshold_map50}")
            if agg.threshold_recall is not None:
                parts.append(f"recall ≥ {agg.threshold_recall}")
            thresh_str = " / ".join(parts) if parts else "—"

            if agg.passes_threshold is None:
                result_str = "—"
            elif agg.passes_threshold:
                result_str = "PASS ✓"
            else:
                result_str = "FAIL ✗"

            print(
                f"  {name:<12} {map50_str:<26} {rec_str:<24} {thresh_str:<14} {result_str}"
            )

        print()
        gate = "PASSED" if report.quality_gate_passed else "FAILED"
        print(f"[evaluate] Quality gate: {gate}")
        if report.threshold_violations:
            for v in report.threshold_violations:
                print(f"[evaluate]   ✗ {v}")
