"""Feeder driver interface and base implementations."""

from abc import ABC, abstractmethod


class FeederDriver(ABC):
    """Abstract feeder driver interface."""

    @abstractmethod
    async def activate(self) -> bool:
        """Activate the feeder.

        Returns:
            True if activation succeeded, False otherwise.
        """
        ...

    @abstractmethod
    async def deactivate(self) -> bool:
        """Deactivate the feeder.

        Returns:
            True if deactivation succeeded, False otherwise.
        """
        ...

    @abstractmethod
    async def is_active(self) -> bool:
        """Check if feeder is currently active."""
        ...
