from __future__ import annotations

from typing import Any

import numpy as np

from labvla.env import LabObservation


class ControlExecutionError(Exception):
    """Raised when a scripted control attempt fails and may be retried."""

    def __init__(
        self,
        message: str,
        *,
        attempt: int = 1,
        actions: list[np.ndarray] | None = None,
        observations: list[LabObservation] | None = None,
        frames: list[np.ndarray] | None = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempt = attempt
        self.actions = actions or []
        self.observations = observations or []
        self.frames = frames or []
        self.info = info or {}
