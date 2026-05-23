"""Simulated feeder driver for development and testing."""

from pi_controller.drivers.base import FeederDriver


class SimulatedFeederDriver(FeederDriver):
    """Simulated feeder driver that tracks state in memory."""

    def __init__(self) -> None:
        """Initialize the simulated driver."""
        self._active = False

    async def activate(self) -> bool:
        """Activate the feeder (simulated)."""
        self._active = True
        return True

    async def deactivate(self) -> bool:
        """Deactivate the feeder (simulated)."""
        self._active = False
        return True

    async def is_active(self) -> bool:
        """Check if feeder is active."""
        return self._active
