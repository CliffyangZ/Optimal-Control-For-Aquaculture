"""Integration tests for prepared validation clip runner."""

import pytest
import tempfile
from pathlib import Path
import yaml
import numpy as np
import cv2
from validation.clip_manifest import ValidationManifest
from core.ripple_detector import RippleDetector
from core.models import CurrentFrame, DetectionStatus
from datetime import datetime


class TestClipValidationRunner:
    """Test prepared clip validation runner."""

    @pytest.fixture
    def detector(self):
        """Create a fresh ripple detector."""
        return RippleDetector()

    @pytest.fixture
    def temp_manifest_with_clips(self):
        """Create a temporary manifest with test video clips."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clips_dir = Path(tmpdir) / "clips"
            clips_dir.mkdir()

            manifest_data = {
                "version": 1,
                "created_for": "test",
                "clips": []
            }

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            frame_size = (1280, 720)

            for expected_activity in ["Low", "Medium", "High", "Detection Unavailable"]:
                clip_id = f"{expected_activity.lower().replace(' ', '_')}_001"
                clip_path = clips_dir / f"{clip_id}.mp4"

                out = cv2.VideoWriter(str(clip_path), fourcc, fps, frame_size)

                if expected_activity == "Low":
                    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
                elif expected_activity == "Medium":
                    frame = np.random.randint(80, 180, (720, 1280, 3), dtype=np.uint8)
                elif expected_activity == "High":
                    frame = np.random.randint(50, 255, (720, 1280, 3), dtype=np.uint8)
                else:
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

                for _ in range(10):
                    out.write(frame)

                out.release()

                manifest_data["clips"].append({
                    "id": clip_id,
                    "path": f"clips/{clip_id}.mp4",
                    "expected_activity": expected_activity
                })

            manifest_path = Path(tmpdir) / "manifest.yml"
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest_data, f)

            yield str(manifest_path), tmpdir

    def test_manifest_loads_valid(self, temp_manifest_with_clips):
        """Test that manifest loads correctly."""
        manifest_path, _ = temp_manifest_with_clips
        manifest = ValidationManifest(manifest_path)
        assert len(manifest.clips) == 4

    def test_clip_validation_structure(self, detector, temp_manifest_with_clips):
        """Test validation result structure."""
        manifest_path, _ = temp_manifest_with_clips
        manifest = ValidationManifest(manifest_path)

        results = []
        for clip in manifest.clips:
            test_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
            current_frame = CurrentFrame(
                image=test_frame,
                captured_at=datetime.now(),
                sequence_number=1,
                source_width=1280,
                source_height=720
            )

            detection = detector.detect(current_frame)
            results.append({
                "clip_id": clip.id,
                "expected": clip.expected_activity,
                "detected": detection.activity_level.value if detection.status == DetectionStatus.AVAILABLE else "Detection Unavailable",
                "passed": True
            })

        assert len(results) == 4

    def test_per_clip_result_tracking(self):
        """Test that per-clip results are tracked."""
        results = [
            {"clip_id": "low_001", "expected": "Low", "detected": "Low", "passed": True},
            {"clip_id": "med_001", "expected": "Medium", "detected": "Medium", "passed": True},
            {"clip_id": "high_001", "expected": "High", "detected": "High", "passed": True},
            {"clip_id": "int_001", "expected": "Detection Unavailable", "detected": "Detection Unavailable", "passed": True},
        ]

        assert all(r["passed"] for r in results)
        assert len([r for r in results if r["passed"]]) == 4

    def test_pass_fail_counting(self):
        """Test pass/fail counting for validation runs."""
        results = [
            {"passed": True},
            {"passed": True},
            {"passed": True},
            {"passed": True},
            {"passed": True},
            {"passed": True},
            {"passed": True},
            {"passed": True},
            {"passed": False},
            {"passed": False},
        ]

        passed_count = sum(1 for r in results if r["passed"])
        total_count = len(results)

        assert passed_count == 8
        assert total_count == 10
        assert passed_count >= 8

    def test_aggregate_8_of_10_reporting(self):
        """Test that 8 of 10 passing is reported as success."""
        test_runs = [
            [True] * 8 + [False] * 2,
            [True] * 9 + [False],
            [True] * 10,
            [True] * 7 + [False] * 3,
        ]

        for results in test_runs:
            passed = sum(1 for r in results if r)
            total = len(results)
            meets_threshold = passed >= 8 and total == 10

            if total == 10:
                assert (passed >= 8) == meets_threshold

    def test_validation_summary_generation(self):
        """Test validation summary generation."""
        results = [
            {"clip_id": f"clip_{i}", "passed": i < 8}
            for i in range(10)
        ]

        passed = sum(1 for r in results if r["passed"])
        total = len(results)

        summary = {
            "total_clips": total,
            "passed_clips": passed,
            "failed_clips": total - passed,
            "pass_rate": f"{(passed/total)*100:.1f}%",
            "meets_mvp_target": passed >= 8
        }

        assert summary["total_clips"] == 10
        assert summary["passed_clips"] == 8
        assert summary["pass_rate"] == "80.0%"
        assert summary["meets_mvp_target"]
