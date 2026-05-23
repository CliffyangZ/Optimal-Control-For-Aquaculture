"""Unit tests for `core.train.evaluator`.

Covers:
- quality gate passes when all class metrics exceed thresholds
- quality gate fails and populates threshold_violations correctly
- evaluation_report.json schema (no GPU required — all metrics are mocked)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import ClassThreshold, EvaluationConfig
from core.train.evaluator import (
    ClassMetrics,
    Evaluator,
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


def _make_report(class_values: dict[str, dict[str, float]]) -> MetricsReport:
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

    # Compute quality gate with default config
    config = _default_config()
    violations: list[str] = []
    for name in ("fish", "feed", "debris", "lumpfish"):
        cm = classes[name]
        threshold_cfg = config.thresholds.get(name)
        if not threshold_cfg:
            continue
        if threshold_cfg.map50 is not None and cm.map50 < threshold_cfg.map50:
            violations.append(f"{name}: map50={cm.map50:.3f} < threshold {threshold_cfg.map50}")
        if threshold_cfg.recall is not None and cm.recall < threshold_cfg.recall:
            violations.append(f"{name}: recall={cm.recall:.3f} < threshold {threshold_cfg.recall}")

    return MetricsReport(
        classes=classes,
        quality_gate_passed=len(violations) == 0,
        threshold_violations=violations,
    )


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
    report = _make_report({"fish": {"map50": 0.80}, "feed": {"map50": 0.75, "recall": 0.80}})
    assert report.quality_gate_passed is True
    assert report.threshold_violations == []


def test_quality_gate_fails_on_low_map50() -> None:
    # fish map50=0.70 < threshold 0.75
    report = _make_report({"fish": {"map50": 0.70}, "feed": {"map50": 0.75, "recall": 0.80}})
    assert report.quality_gate_passed is False
    assert any("fish" in v and "map50" in v for v in report.threshold_violations)


def test_quality_gate_fails_on_low_recall() -> None:
    # feed recall=0.70 < threshold 0.75
    report = _make_report({"fish": {"map50": 0.80}, "feed": {"map50": 0.75, "recall": 0.70}})
    assert report.quality_gate_passed is False
    assert any("feed" in v and "recall" in v for v in report.threshold_violations)


def test_quality_gate_multiple_violations() -> None:
    report = _make_report({
        "fish": {"map50": 0.60},
        "feed": {"map50": 0.60, "recall": 0.60},
    })
    assert report.quality_gate_passed is False
    assert len(report.threshold_violations) >= 3  # fish map50, feed map50, feed recall


def test_no_threshold_classes_do_not_affect_gate() -> None:
    """debris and lumpfish have no thresholds — gate should be indifferent to them."""
    report = _make_report({
        "fish": {"map50": 0.80},
        "feed": {"map50": 0.75, "recall": 0.80},
        "debris": {"map50": 0.10},
        "lumpfish": {"map50": 0.10},
    })
    assert report.quality_gate_passed is True


# ---------------------------------------------------------------------------
# evaluation_report.json schema
# ---------------------------------------------------------------------------

def test_write_report_json_schema(tmp_path: Path) -> None:
    report = _make_report({"fish": {"map50": 0.80}, "feed": {"map50": 0.75, "recall": 0.80}})
    evaluator = Evaluator(_default_config())

    out = tmp_path / "evaluation_report.json"
    evaluator.write_report(report, out)

    assert out.exists()
    data = json.loads(out.read_text())

    assert "quality_gate_passed" in data
    assert "threshold_violations" in data
    assert "classes" in data

    # Classes must have all 4 classes
    for name in ("fish", "feed", "debris", "lumpfish"):
        assert name in data["classes"]
        cm = data["classes"][name]
        assert "precision" in cm
        assert "recall" in cm
        assert "f1" in cm
        assert "map50" in cm


def test_write_report_json_has_class_fields(tmp_path: Path) -> None:
    report = _make_report({"fish": {"map50": 0.80}, "feed": {"map50": 0.75, "recall": 0.80}})
    evaluator = Evaluator(_default_config())

    out = tmp_path / "evaluation_report.json"
    evaluator.write_report(report, out)
    data = json.loads(out.read_text())

    feed = data["classes"]["feed"]
    assert feed["map50"] == pytest.approx(0.75)
    assert feed["recall"] == pytest.approx(0.80)
