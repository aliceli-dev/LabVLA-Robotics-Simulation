from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from labvla.env import LabEnv, LabObservation
from labvla.vlm import TaskPlan


@dataclass
class ControlResult:
    success: bool
    actions: list[np.ndarray] = field(default_factory=list)
    observations: list[LabObservation] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


class Controller(ABC):
    @abstractmethod
    def execute(self, env: LabEnv, plan: TaskPlan) -> ControlResult:
        raise NotImplementedError
