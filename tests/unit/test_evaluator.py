"""Unit tests for `core.train.evaluator`.

Covers:
- quality gate passes when all class metrics exceed thresholds
- quality gate fails and populates threshold_violations correctly
- mean/std calculation
- cv_report.json schema (no GPU required — all fold metrics are mocked)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import ClassThreshold, EvaluationConfig
from core.train.evaluator import (
    AggregateClassMetrics,
    ClassMetrics,
    CrossFoldEvaluator,
    FoldMetrics,
    MetricsReport,
)


def _make_class_metrics(
    name: str,
    *,
    precision: float = 0.8,
    recall: float = 0.8,
    f1: float = 0.8,
    map50: float = 0.8,
) -> ClassMetrics:
    return ClassMetrics(class_name=name, precision=precision, recall=recall, f1=f1, map50=map50)


def _make_fold(k: int, class_values: dict[str, dict[str, float]]) -> FoldMetrics:
    classes = {}
    for name, vals in class_values.items():
        classes[name] = ClassMetrics(
            class_name=name,
            precision=vals.get("precision", 0.8),
            recall=vals.get("recall", 0.8),
            f1=vals.get("f1", 0.8),
            map50=vals.get("map50", 0.8),
        )
    # Fill in missing classes
    for name in ("fish", "feed", "debris", "lumpfish"):
        if name not in classes:
            classes[name] = _make_class_metrics(name)
    return FoldMetrics(fold_index=k, classes=classes)


def _default_config() -> EvaluationConfig:
    return EvaluationConfig(
        thresholds={
            "feed": ClassThreshold(map50=0.70, recall=0.75),
            "fish": ClassThreshold(map50=0.75),
        }
    )


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def test_quality_gate_passes_when_all_above_threshold() -> None:
    folds = [
        _make_fold(k, {"fish": {"map50": 0.80}, "feed": {"map50": 0.75, "recall": 0.80}})
        for k in range(5)
    ]
    evaluator = CrossFoldEvaluator(_default_config())
    report = evaluator.aggregate(folds)
    assert report.quality_gate_passed is True
    assert report.threshold_violations == []


def test_quality_gate_fails_on_low_map50() -> None:
    # fish map50=0.70 < threshold 0.75
    folds = [
        _make_fold(k, {"fish": {"map50": 0.70}, "feed": {"map50": 0.75, "recall": 0.80}})
        for k in range(5)
    ]
    evaluator = CrossFoldEvaluator(_default_config())
    report = evaluator.aggregate(folds)
    assert report.quality_gate_passed is False
    assert any("fish" in v and "mean_map50" in v for v in report.threshold_violations)


def test_quality_gate_fails_on_low_recall() -> None:
    # feed recall=0.70 < threshold 0.75
    folds = [
        _make_fold(k, {"fish": {"map50": 0.80}, "feed": {"map50": 0.75, "recall": 0.70}})
        for k in range(5)
    ]
    evaluator = CrossFoldEvaluator(_default_config())
    report = evaluator.aggregate(folds)
    assert report.quality_gate_passed is False
    assert any("feed" in v and "recall" in v for v in report.threshold_violations)


def test_quality_gate_multiple_violations() -> None:
    folds = [
        _make_fold(k, {
            "fish": {"map50": 0.60},
            "feed": {"map50": 0.60, "recall": 0.60},
        })
        for k in range(5)
    ]
    evaluator = CrossFoldEvaluator(_default_config())
    report = evaluator.aggregate(folds)
    assert report.quality_gate_passed is False
    assert len(report.threshold_violations) >= 3  # fish map50, feed map50, feed recall


def test_no_threshold_classes_do_not_affect_gate() -> None:
    """debris and lumpfish have no thresholds — gate should be indifferent to them."""
    folds = [
        _make_fold(k, {
            "fish": {"map50": 0.80},
            "feed": {"map50": 0.75, "recall": 0.80},
            "debris": {"map50": 0.10},
            "lumpfish": {"map50": 0.10},
        })
        for k in range(5)
    ]
    evaluator = CrossFoldEvaluator(_default_config())
    report = evaluator.aggregate(folds)
    assert report.quality_gate_passed is True


# ---------------------------------------------------------------------------
# Mean / std calculation
# ---------------------------------------------------------------------------

def test_mean_and_std_calculation() -> None:
    import numpy as np

    map50_values = [0.80, 0.82, 0.78, 0.81, 0.79]
    folds = [
        _make_fold(k, {"fish": {"map50": v}, "feed": {"map50": 0.75, "recall": 0.80}})
        for k, v in enumerate(map50_values)
    ]
    evaluator = CrossFoldEvaluator(_default_config())
    report = evaluator.aggregate(folds)

    agg = report.aggregate["fish"]
    assert abs(agg.mean_map50 - float(np.mean(map50_values))) < 1e-6
    assert abs(agg.std_map50 - float(np.std(map50_values))) < 1e-6


# ---------------------------------------------------------------------------
# cv_report.json schema
# ---------------------------------------------------------------------------

def test_write_report_json_schema(tmp_path: Path) -> None:
    folds = [
        _make_fold(k, {"fish": {"map50": 0.80}, "feed": {"map50": 0.75, "recall": 0.80}})
        for k in range(5)
    ]
    evaluator = CrossFoldEvaluator(_default_config())
    report = evaluator.aggregate(folds)

    out = tmp_path / "cv_report.json"
    evaluator.write_report(report, out)

    assert out.exists()
    data = json.loads(out.read_text())

    assert "quality_gate_passed" in data
    assert "threshold_violations" in data
    assert "aggregate" in data
    assert "fold_metrics" in data

    # Aggregate must have all 4 classes
    for name in ("fish", "feed", "debris", "lumpfish"):
        assert name in data["aggregate"]
        agg = data["aggregate"][name]
        assert "mean_map50" in agg
        assert "std_map50" in agg
        assert "mean_recall" in agg


def test_write_report_json_has_feed_aggregate_fields(tmp_path: Path) -> None:
    folds = [
        _make_fold(k, {"fish": {"map50": 0.80}, "feed": {"map50": 0.75, "recall": 0.80}})
        for k in range(5)
    ]
    evaluator = CrossFoldEvaluator(_default_config())
    report = evaluator.aggregate(folds)

    out = tmp_path / "cv_report.json"
    evaluator.write_report(report, out)
    data = json.loads(out.read_text())

    feed = data["aggregate"]["feed"]
    assert feed["threshold_map50"] == pytest.approx(0.70)
    assert feed["threshold_recall"] == pytest.approx(0.75)


def test_write_report_fold_metrics_count(tmp_path: Path) -> None:
    folds = [_make_fold(k, {}) for k in range(5)]
    evaluator = CrossFoldEvaluator(_default_config())
    report = evaluator.aggregate(folds)
    out = tmp_path / "cv_report.json"
    evaluator.write_report(report, out)
    data = json.loads(out.read_text())
    assert len(data["fold_metrics"]) == 5
