"""Raspberry Pi GPIO feeder driver for hardware mode."""

from pi_controller.config import config
from pi_controller.drivers.base import FeederDriver


class GPIOFeederDriver(FeederDriver):
    """GPIO-based feeder driver for Raspberry Pi hardware."""

    def __init__(self) -> None:
        """Initialize the GPIO driver."""
        self._active = False
        self._gpio_module = None
        self._setup_gpio()

    def _setup_gpio(self) -> None:
        """Set up GPIO pins (lazy import for optional dependency)."""
        try:
            import RPi.GPIO as GPIO

            self._gpio_module = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(config.GPIO_OUTPUT_CHANNEL, GPIO.OUT)
            GPIO.output(config.GPIO_OUTPUT_CHANNEL, GPIO.LOW)
        except ImportError:
            raise RuntimeError("RPi.GPIO not installed; hardware mode unavailable")
        except Exception as e:
            raise RuntimeError(f"GPIO setup failed: {e}")

    async def activate(self) -> bool:
        """Activate the feeder via GPIO."""
        if not self._gpio_module:
            return False
        try:
            level = self._gpio_module.HIGH if config.GPIO_ACTIVE_HIGH else self._gpio_module.LOW
            self._gpio_module.output(config.GPIO_OUTPUT_CHANNEL, level)
            self._active = True
            return True
        except Exception:
            return False

    async def deactivate(self) -> bool:
        """Deactivate the feeder via GPIO."""
        if not self._gpio_module:
            return False
        try:
            level = self._gpio_module.LOW if config.GPIO_ACTIVE_HIGH else self._gpio_module.HIGH
            self._gpio_module.output(config.GPIO_OUTPUT_CHANNEL, level)
            self._active = False
            return True
        except Exception:
            return False

    async def is_active(self) -> bool:
        """Check if feeder is active."""
        return self._active

    def cleanup(self) -> None:
        """Clean up GPIO on shutdown."""
        if self._gpio_module:
            self._gpio_module.cleanup(config.GPIO_OUTPUT_CHANNEL)
