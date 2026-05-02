"""Integration tests for validation manifest parsing and validation."""

import pytest
import tempfile
from pathlib import Path
import yaml
from validation.clip_manifest import ValidationManifest, ValidationClip


class TestValidationManifestParsing:
    """Test manifest loading and parsing."""

    @pytest.fixture
    def temp_manifest(self):
        """Create a temporary manifest directory and file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_data = {
                "version": 1,
                "created_for": "test",
                "clips": [
                    {
                        "id": "low-001",
                        "path": "low-001.mp4",
                        "expected_activity": "Low",
                        "notes": "Calm water"
                    },
                    {
                        "id": "high-001",
                        "path": "high-001.mp4",
                        "expected_activity": "High",
                        "interference_labels": []
                    }
                ]
            }
            manifest_path = Path(tmpdir) / "manifest.yml"
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest_data, f)

            yield str(manifest_path), tmpdir

    def test_load_valid_manifest(self, temp_manifest):
        """Test loading a valid manifest."""
        manifest_path, _ = temp_manifest
        manifest = ValidationManifest(manifest_path)
        assert len(manifest.clips) == 2
        assert manifest.clips[0].id == "low-001"
        assert manifest.clips[0].expected_activity == "Low"

    def test_missing_manifest_file(self):
        """Test that missing manifest raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ValidationManifest("nonexistent/manifest.yml")

    def test_invalid_expected_activity(self):
        """Test that invalid activity values are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_data = {
                "clips": [
                    {
                        "id": "bad-001",
                        "path": "bad.mp4",
                        "expected_activity": "InvalidActivity"
                    }
                ]
            }
            manifest_path = Path(tmpdir) / "manifest.yml"
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest_data, f)

            with pytest.raises(ValueError):
                ValidationManifest(str(manifest_path))

    def test_duplicate_clip_ids(self):
        """Test that duplicate clip IDs are detected during validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_data = {
                "clips": [
                    {"id": "same-id", "path": "clip1.mp4", "expected_activity": "Low"},
                    {"id": "same-id", "path": "clip2.mp4", "expected_activity": "Low"}
                ]
            }
            manifest_path = Path(tmpdir) / "manifest.yml"
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest_data, f)

            manifest = ValidationManifest(str(manifest_path))
            validation_result = manifest.validate_structure()
            assert not validation_result["valid"]
            assert any("Duplicate clip ID" in error for error in validation_result["errors"])


class TestManifestValidation:
    """Test manifest structure validation."""

    def test_missing_low_clips(self):
        """Test that missing Low activity clips is detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_data = {
                "clips": [
                    {"id": "low-001", "path": "low1.mp4", "expected_activity": "Low"},
                    {"id": "med-001", "path": "med1.mp4", "expected_activity": "Medium"},
                    {"id": "med-002", "path": "med2.mp4", "expected_activity": "Medium"},
                    {"id": "med-003", "path": "med3.mp4", "expected_activity": "Medium"},
                    {"id": "high-001", "path": "high1.mp4", "expected_activity": "High"},
                    {"id": "high-002", "path": "high2.mp4", "expected_activity": "High"},
                    {"id": "high-003", "path": "high3.mp4", "expected_activity": "High"},
                    {"id": "int-001", "path": "int1.mp4", "expected_activity": "Detection Unavailable"}
                ]
            }
            manifest_path = Path(tmpdir) / "manifest.yml"
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest_data, f)

            manifest = ValidationManifest(str(manifest_path))
            result = manifest.validate_structure()
            assert not result["valid"]
            assert result["summary"]["low_count"] == 1

    def test_missing_required_clips_count(self):
        """Test various missing clip scenarios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_data = {
                "clips": [
                    {"id": "low-001", "path": "low1.mp4", "expected_activity": "Low"},
                    {"id": "low-002", "path": "low2.mp4", "expected_activity": "Low"},
                    {"id": "low-003", "path": "low3.mp4", "expected_activity": "Low"},
                    {"id": "med-001", "path": "med1.mp4", "expected_activity": "Medium"},
                    {"id": "med-002", "path": "med2.mp4", "expected_activity": "Medium"},
                    {"id": "med-003", "path": "med3.mp4", "expected_activity": "Medium"},
                    {"id": "high-001", "path": "high1.mp4", "expected_activity": "High"},
                    {"id": "high-002", "path": "high2.mp4", "expected_activity": "High"},
                    {"id": "high-003", "path": "high3.mp4", "expected_activity": "High"},
                    {"id": "int-001", "path": "int1.mp4", "expected_activity": "Detection Unavailable"}
                ]
            }
            manifest_path = Path(tmpdir) / "manifest.yml"
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest_data, f)

            manifest = ValidationManifest(str(manifest_path))
            result = manifest.validate_structure()
            assert result["valid"]
            assert result["summary"]["total_clips"] == 10


class TestMissingClipDetection:
    """Test detection of missing clip files."""

    def test_missing_clip_files(self):
        """Test that missing files are reported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_data = {
                "clips": [
                    {"id": "exists", "path": "exists.mp4", "expected_activity": "Low"},
                    {"id": "missing", "path": "missing.mp4", "expected_activity": "Low"}
                ]
            }
            manifest_path = Path(tmpdir) / "manifest.yml"
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest_data, f)

            Path(tmpdir, "exists.mp4").touch()

            manifest = ValidationManifest(str(manifest_path))
            missing_result = manifest.resolve_paths()
            assert missing_result["missing_count"] == 1
            assert missing_result["missing_clips"][0]["id"] == "missing"

    def test_relative_path_resolution(self):
        """Test that relative paths are resolved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clips_dir = Path(tmpdir) / "clips"
            clips_dir.mkdir()

            manifest_data = {
                "clips": [
                    {"id": "clip1", "path": "clips/video.mp4", "expected_activity": "Low"}
                ]
            }
            manifest_path = Path(tmpdir) / "manifest.yml"
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest_data, f)

            (clips_dir / "video.mp4").touch()

            manifest = ValidationManifest(str(manifest_path))
            missing_result = manifest.resolve_paths()
            assert missing_result["missing_count"] == 0
