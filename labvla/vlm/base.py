from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class TaskPlan:
    object: str
    destination: str
    action: str = "pick_and_place"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VLMBackend(ABC):
    @abstractmethod
    def plan(self, instruction: str, image: np.ndarray) -> TaskPlan:
        raise NotImplementedError
