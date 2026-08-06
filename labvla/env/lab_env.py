from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LabState:
    object_positions: dict[str, np.ndarray]
    gripper_open: float
    ee_position: np.ndarray
    success: bool = False

    def to_vector(self, object_order: list[str] | None = None) -> np.ndarray:
        order = object_order or sorted(self.object_positions.keys())
        parts: list[np.ndarray] = []
        for name in order:
            pos = self.object_positions.get(name, np.zeros(3, dtype=np.float32))
            parts.append(np.asarray(pos[:2], dtype=np.float32))
        parts.append(np.asarray(self.ee_position[:2], dtype=np.float32))
        parts.append(np.array([self.gripper_open, float(self.success)], dtype=np.float32))
        return np.concatenate(parts, axis=0)


@dataclass
class LabObservation:
    image: np.ndarray
    state: LabState
    info: dict[str, Any] = field(default_factory=dict)


class LabEnv:
    def __init__(
        self,
        robot: str = "panda",
        objects: list[str] | None = None,
        racks: list[str] | None = None,
        seed: int = 0,
    ) -> None:
        self.robot = robot
        self.objects = objects or ["red_tube", "blue_tube"]
        self.racks = racks or ["rack_a", "rack_b"]
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._state = self._initial_state()

    def _initial_state(self) -> LabState:
        positions = {
            "red_tube": np.array([0.15, 0.10, 0.02], dtype=np.float32),
            "blue_tube": np.array([0.15, -0.10, 0.02], dtype=np.float32),
            "rack_a": np.array([-0.20, 0.10, 0.00], dtype=np.float32),
            "rack_b": np.array([-0.20, -0.10, 0.00], dtype=np.float32),
        }
        return LabState(
            object_positions=positions,
            gripper_open=1.0,
            ee_position=np.array([0.0, 0.0, 0.20], dtype=np.float32),
            success=False,
        )

    def reset(self) -> LabObservation:
        self._rng = np.random.default_rng(self.seed)
        self._state = self._initial_state()
        return self._observe()

    def _observe(self) -> LabObservation:
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[:, :, 0] = 40
        image[:, :, 1] = 40
        image[:, :, 2] = 48
        red = self._state.object_positions["red_tube"]
        blue = self._state.object_positions["blue_tube"]
        self._paint_dot(image, red, (220, 40, 40))
        self._paint_dot(image, blue, (40, 80, 220))
        return LabObservation(image=image, state=self._state, info={"robot": self.robot})

    def _paint_dot(self, image: np.ndarray, pos: np.ndarray, color: tuple[int, int, int]) -> None:
        h, w = image.shape[:2]
        u = int((pos[0] + 0.35) / 0.70 * (w - 1))
        v = int((0.35 - pos[1]) / 0.70 * (h - 1))
        u = int(np.clip(u, 2, w - 3))
        v = int(np.clip(v, 2, h - 3))
        image[v - 2 : v + 3, u - 2 : u + 3] = color

    def step(self, action: np.ndarray) -> LabObservation:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        delta = action[:3] if action.shape[0] >= 3 else np.zeros(3, dtype=np.float32)
        grip = float(action[3]) if action.shape[0] >= 4 else self._state.gripper_open
        self._state.ee_position = self._state.ee_position + delta
        self._state.gripper_open = float(np.clip(grip, 0.0, 1.0))
        return self._observe()

    def apply_placement(self, object_name: str, destination: str) -> LabObservation:
        if object_name not in self._state.object_positions:
            raise KeyError(object_name)
        if destination not in self._state.object_positions:
            raise KeyError(destination)
        target = self._state.object_positions[destination].copy()
        target[2] = 0.02
        self._state.object_positions[object_name] = target
        self._state.ee_position = target + np.array([0.0, 0.0, 0.12], dtype=np.float32)
        self._state.gripper_open = 1.0
        self._state.success = True
        return self._observe()

    @property
    def state(self) -> LabState:
        return self._state
