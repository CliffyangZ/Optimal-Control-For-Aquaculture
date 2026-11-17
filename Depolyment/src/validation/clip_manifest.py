"""Validation manifest parsing and validation."""

import yaml
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ValidationClip:
    """Represents a validation clip entry."""
    id: str
    path: str
    expected_activity: str
    interference_labels: List[str] = None
    notes: str = None

    def __post_init__(self):
        if self.interference_labels is None:
            self.interference_labels = []


class ValidationManifest:
    """Loads and validates a validation manifest file."""

    REQUIRED_LOW_COUNT = 3
    REQUIRED_MEDIUM_COUNT = 3
    REQUIRED_HIGH_COUNT = 3
    REQUIRED_INTERFERENCE_COUNT = 1
    VALID_ACTIVITIES = {"Low", "Medium", "High", "Detection Unavailable"}

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        self.clips: List[ValidationClip] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load and parse the manifest YAML file."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        with open(self.manifest_path, 'r') as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        clips_data = data.get('clips', [])
        if not isinstance(clips_data, list):
            raise ValueError("'clips' must be a list")

        for clip_data in clips_data:
            clip = self._parse_clip(clip_data)
            self.clips.append(clip)

    def _parse_clip(self, clip_data: dict) -> ValidationClip:
        """Parse and validate a single clip entry."""
        required_fields = {'id', 'path', 'expected_activity'}
        if not isinstance(clip_data, dict):
            raise ValueError("Each clip must be a dictionary")

        if not required_fields.issubset(clip_data.keys()):
            raise ValueError(f"Clip missing required fields: {required_fields}")

        clip_id = clip_data['id']
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ValueError("Clip 'id' must be a non-empty string")

        path = clip_data['path']
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"Clip {clip_id}: 'path' must be a non-empty string")

        activity = clip_data['expected_activity']
        if activity not in self.VALID_ACTIVITIES:
            raise ValueError(
                f"Clip {clip_id}: 'expected_activity' must be one of {self.VALID_ACTIVITIES}"
            )

        interference = clip_data.get('interference_labels', [])
        if not isinstance(interference, list):
            raise ValueError(f"Clip {clip_id}: 'interference_labels' must be a list")

        notes = clip_data.get('notes', '')
        if not isinstance(notes, str):
            raise ValueError(f"Clip {clip_id}: 'notes' must be a string")

        return ValidationClip(
            id=clip_id,
            path=path,
            expected_activity=activity,
            interference_labels=interference,
            notes=notes
        )

    def validate_structure(self) -> dict:
        """Validate manifest structure against requirements."""
        errors = []

        low_count = sum(1 for c in self.clips if c.expected_activity == "Low")
        medium_count = sum(1 for c in self.clips if c.expected_activity == "Medium")
        high_count = sum(1 for c in self.clips if c.expected_activity == "High")
        interference_count = sum(
            1 for c in self.clips
            if c.expected_activity == "Detection Unavailable" or c.interference_labels
        )

        if low_count < self.REQUIRED_LOW_COUNT:
            errors.append(
                f"Need at least {self.REQUIRED_LOW_COUNT} Low clips, found {low_count}"
            )
        if medium_count < self.REQUIRED_MEDIUM_COUNT:
            errors.append(
                f"Need at least {self.REQUIRED_MEDIUM_COUNT} Medium clips, found {medium_count}"
            )
        if high_count < self.REQUIRED_HIGH_COUNT:
            errors.append(
                f"Need at least {self.REQUIRED_HIGH_COUNT} High clips, found {high_count}"
            )
        if interference_count < self.REQUIRED_INTERFERENCE_COUNT:
            errors.append(
                f"Need at least {self.REQUIRED_INTERFERENCE_COUNT} interference clip, found {interference_count}"
            )

        seen_ids = set()
        for clip in self.clips:
            if clip.id in seen_ids:
                errors.append(f"Duplicate clip ID: {clip.id}")
            seen_ids.add(clip.id)

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "summary": {
                "total_clips": len(self.clips),
                "low_count": low_count,
                "medium_count": medium_count,
                "high_count": high_count,
                "interference_count": interference_count,
            }
        }

    def resolve_paths(self) -> dict:
        """Resolve relative paths and check for missing files."""
        missing = []
        for clip in self.clips:
            clip_path = Path(clip.path)
            if not clip_path.is_absolute():
                clip_path = self.manifest_path.parent / clip_path

            if not clip_path.exists():
                missing.append({
                    "id": clip.id,
                    "original_path": clip.path,
                    "resolved_path": str(clip_path)
                })

        return {
            "missing_count": len(missing),
            "missing_clips": missing
        }
