"""Runtime configuration for the Pi controller."""

import os
from pathlib import Path
from typing import Literal

class Config:
    """Controller configuration with environment overrides."""

    # Identity
    CONTROLLER_ID: str = os.getenv("PI_CONTROLLER_ID", "pi-controller-1")

    # Storage
    STORAGE_DIR: Path = Path(os.getenv("PI_CONTROLLER_STORAGE_DIR", "./data/controller"))
    DATABASE_PATH: Path = STORAGE_DIR / "controller.db"

    # Driver mode: simulated or hardware
    DRIVER_MODE: Literal["simulated", "hardware"] = os.getenv(
        "PI_CONTROLLER_DRIVER_MODE", "simulated"
    )

    # Duration limits (seconds)
    DEFAULT_SCHEDULED_DURATION_SECONDS: int = int(
        os.getenv("PI_CONTROLLER_DEFAULT_SCHEDULED_DURATION", "10")
    )
    MAX_SCHEDULED_DURATION_SECONDS: int = int(
        os.getenv("PI_CONTROLLER_MAX_SCHEDULED_DURATION", "60")
    )
    DEFAULT_MANUAL_DURATION_SECONDS: int = int(
        os.getenv("PI_CONTROLLER_DEFAULT_MANUAL_DURATION", "10")
    )
    MAX_MANUAL_DURATION_SECONDS: int = int(
        os.getenv("PI_CONTROLLER_MAX_MANUAL_DURATION", "60")
    )

    # Scheduler
    SCHEDULE_TICK_SECONDS: int = int(os.getenv("PI_CONTROLLER_SCHEDULE_TICK_SECONDS", "5"))

    # Missed runs and history
    MISSED_RUN_LOOKBACK_HOURS: int = 24
    FEED_RUN_RETENTION_COUNT: int = 1000

    # GPIO configuration (hardware mode only)
    GPIO_OUTPUT_CHANNEL: int = int(os.getenv("PI_CONTROLLER_GPIO_CHANNEL", "17"))
    GPIO_ACTIVE_HIGH: bool = os.getenv("PI_CONTROLLER_GPIO_ACTIVE_HIGH", "true").lower() == "true"

    # Network
    LOCAL_TRUSTED_NETWORK_ONLY: bool = True

    # Ensure storage directory exists
    @classmethod
    def ensure_storage(cls) -> None:
        """Create storage directory if it doesn't exist."""
        cls.STORAGE_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
