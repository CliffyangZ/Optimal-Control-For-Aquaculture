"""Tests for activity level mapping and threshold validation."""

import pytest
from core.activity_levels import ActivityLevel, get_activity_level, validate_thresholds
from config.setting import ACTIVITY_THRESHOLDS


class TestActivityLevelMapping:
    """Test activity score to level mapping."""

    def test_low_activity_range(self):
        """Test Low activity level mapping."""
        assert get_activity_level(0) == ActivityLevel.LOW
        assert get_activity_level(16) == ActivityLevel.LOW
        assert get_activity_level(33) == ActivityLevel.LOW

    def test_medium_activity_range(self):
        """Test Medium activity level mapping."""
        assert get_activity_level(34) == ActivityLevel.MEDIUM
        assert get_activity_level(50) == ActivityLevel.MEDIUM
        assert get_activity_level(66) == ActivityLevel.MEDIUM

    def test_high_activity_range(self):
        """Test High activity level mapping."""
        assert get_activity_level(67) == ActivityLevel.HIGH
        assert get_activity_level(84) == ActivityLevel.HIGH
        assert get_activity_level(100) == ActivityLevel.HIGH

    def test_boundary_transitions(self):
        """Test boundaries between activity levels."""
        assert get_activity_level(33) == ActivityLevel.LOW
        assert get_activity_level(34) == ActivityLevel.MEDIUM
        assert get_activity_level(66) == ActivityLevel.MEDIUM
        assert get_activity_level(67) == ActivityLevel.HIGH

    def test_all_scores_mapped(self):
        """Verify all scores 0-100 map to an activity level."""
        for score in range(101):
            level = get_activity_level(score)
            assert level in ActivityLevel

    def test_invalid_scores(self):
        """Test that invalid scores raise errors."""
        with pytest.raises(ValueError):
            get_activity_level(-1)
        with pytest.raises(ValueError):
            get_activity_level(101)
        with pytest.raises(ValueError):
            get_activity_level(50.5)


class TestThresholdValidation:
    """Test threshold configuration validation."""

    def test_valid_thresholds(self):
        """Test that configured thresholds are valid."""
        validate_thresholds(ACTIVITY_THRESHOLDS)

    def test_missing_level_key(self):
        """Test that missing threshold keys are rejected."""
        invalid = {"low": [0, 33], "medium": [34, 66]}
        with pytest.raises(ValueError):
            validate_thresholds(invalid)

    def test_invalid_range_format(self):
        """Test that invalid range formats are rejected."""
        invalid = {"low": [0], "medium": [34, 66], "high": [67, 100]}
        with pytest.raises(ValueError):
            validate_thresholds(invalid)

    def test_negative_lower_bound(self):
        """Test that negative lower bounds are rejected."""
        invalid = {"low": [-1, 33], "medium": [34, 66], "high": [67, 100]}
        with pytest.raises(ValueError):
            validate_thresholds(invalid)

    def test_upper_bound_exceeds_100(self):
        """Test that upper bounds exceeding 100 are rejected."""
        invalid = {"low": [0, 33], "medium": [34, 66], "high": [67, 101]}
        with pytest.raises(ValueError):
            validate_thresholds(invalid)

    def test_inverted_range(self):
        """Test that inverted ranges are rejected."""
        invalid = {"low": [33, 0], "medium": [34, 66], "high": [67, 100]}
        with pytest.raises(ValueError):
            validate_thresholds(invalid)

    def test_gap_in_coverage(self):
        """Test that gaps in score coverage are detected."""
        invalid = {"low": [0, 32], "medium": [34, 66], "high": [67, 100]}
        with pytest.raises(ValueError):
            validate_thresholds(invalid)

    def test_overlapping_ranges(self):
        """Test that overlapping ranges are detected."""
        invalid = {"low": [0, 34], "medium": [34, 66], "high": [67, 100]}
        with pytest.raises(ValueError):
            validate_thresholds(invalid)

    def test_non_dict_input(self):
        """Test that non-dict input is rejected."""
        with pytest.raises(ValueError):
            validate_thresholds([0, 33, 34, 66, 67, 100])
