"""Validation rules for schedules, durations, and operations."""

import re
from typing import Optional

from pi_controller.config import config
from pi_controller.domain.models import Schedule, Weekday


class ValidationError(Exception):
    """Validation error with field and detail context."""

    def __init__(
        self,
        error_code: str,
        message: str,
        field: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        self.error_code = error_code
        self.message = message
        self.field = field
        self.details = details or {}
        super().__init__(message)


def validate_schedule_name(name: str) -> None:
    """Validate schedule name is not blank."""
    if not name or not name.strip():
        raise ValidationError(
            "invalid_schedule",
            "Schedule name is required.",
            field="name",
        )


def validate_weekdays(weekdays: list[Weekday]) -> None:
    """Validate at least one weekday is selected."""
    if not weekdays:
        raise ValidationError(
            "invalid_schedule",
            "At least one weekday is required.",
            field="weekdays",
        )
    if len(weekdays) != len(set(weekdays)):
        raise ValidationError(
            "invalid_schedule",
            "Duplicate weekdays are not allowed.",
            field="weekdays",
        )


def validate_time_format(time_str: str) -> None:
    """Validate HH:MM format and valid 24-hour time."""
    if not re.match(r"^([0-1][0-9]|2[0-3]):[0-5][0-9]$", time_str):
        raise ValidationError(
            "invalid_schedule",
            "Time must be in HH:MM format (00:00 to 23:59).",
            field="times",
            details={"time": time_str},
        )


def validate_times(times: list[str]) -> None:
    """Validate times are unique and valid."""
    if not times:
        raise ValidationError(
            "invalid_schedule",
            "At least one feeding time is required.",
            field="times",
        )
    for t in times:
        validate_time_format(t)
    if len(times) != len(set(times)):
        raise ValidationError(
            "invalid_schedule",
            "Duplicate times within a schedule are not allowed.",
            field="times",
        )


def validate_duration(
    duration_seconds: int,
    max_duration: int = config.MAX_SCHEDULED_DURATION_SECONDS,
) -> None:
    """Validate duration is positive and within limits."""
    if duration_seconds <= 0:
        raise ValidationError(
            "invalid_duration",
            "Duration must be positive.",
            field="duration_seconds",
            details={"duration_seconds": duration_seconds},
        )
    if duration_seconds > max_duration:
        raise ValidationError(
            "invalid_duration",
            f"Duration must not exceed {max_duration} seconds.",
            field="duration_seconds",
            details={"max": max_duration, "duration_seconds": duration_seconds},
        )


def validate_manual_duration(duration_seconds: int) -> None:
    """Validate manual feed duration."""
    validate_duration(
        duration_seconds,
        max_duration=config.MAX_MANUAL_DURATION_SECONDS,
    )


def validate_schedule_duration(duration_seconds: int) -> None:
    """Validate scheduled feed duration."""
    validate_duration(
        duration_seconds,
        max_duration=config.MAX_SCHEDULED_DURATION_SECONDS,
    )


def validate_create_request(
    name: str,
    weekdays: list[Weekday],
    times: list[str],
    duration_seconds: int,
) -> None:
    """Validate all create request fields."""
    validate_schedule_name(name)
    validate_weekdays(weekdays)
    validate_times(times)
    validate_schedule_duration(duration_seconds)


def validate_update_request(
    name: str,
    weekdays: list[Weekday],
    times: list[str],
    duration_seconds: int,
) -> None:
    """Validate all update request fields."""
    validate_create_request(name, weekdays, times, duration_seconds)


def validate_slot_conflicts(
    new_schedule: Schedule,
    existing_schedules: list[Schedule],
    existing_slots: list[tuple[str, Weekday, str]],
) -> None:
    """Validate new schedule doesn't create duplicate enabled slots.

    existing_slots is a list of (schedule_id, weekday, time) tuples for enabled
    non-deleted slots from other schedules.
    """
    if not new_schedule.enabled:
        return

    existing_slot_set = set(existing_slots)

    for weekday in new_schedule.weekdays:
        for time in new_schedule.times:
            for existing_id, existing_day, existing_time in existing_slot_set:
                if (
                    existing_id != new_schedule.id
                    and existing_day == weekday
                    and existing_time == time
                ):
                    raise ValidationError(
                        "duplicate_feeding_slot",
                        "Enabled schedules cannot share the same weekday and feeding time.",
                        field="times",
                        details={"weekday": weekday.value, "time": time},
                    )


def validate_delete_confirmation(
    schedule_name: str,
    provided_confirmation: str,
) -> None:
    """Validate delete requires schedule name confirmation."""
    if provided_confirmation.strip() != schedule_name.strip():
        raise ValidationError(
            "delete_confirmation_mismatch",
            "Schedule name confirmation does not match.",
            field="schedule_name_confirmation",
            details={"expected": schedule_name, "provided": provided_confirmation},
        )
