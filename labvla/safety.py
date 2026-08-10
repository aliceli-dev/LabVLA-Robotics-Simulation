from __future__ import annotations

from typing import Any

import numpy as np

from labvla.env import LabEnv
from labvla.vlm import TaskPlan

OCCUPANCY_RADIUS = 0.05


class SafetyViolation(Exception):
    def __init__(
        self,
        reason: str,
        message: str,
        *,
        occupant: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.occupant = occupant

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "message": self.message,
            "occupant": self.occupant,
        }


def occupant_at(
    env: LabEnv,
    destination: str,
    *,
    exclude: str | None = None,
    radius: float = OCCUPANCY_RADIUS,
) -> str | None:
    if destination not in env.state.object_positions:
        return None
    dest = env.state.object_positions[destination]
    for name in env.objects:
        if name == exclude:
            continue
        pos = env.state.object_positions[name]
        if float(np.linalg.norm(pos[:2] - dest[:2])) < radius:
            return name
    return None

# Validate a pick-and-place plan before the controller runs.
# I want to check following things:
# 1. object / destination exist
# 2. gripper is open and not already holding something
# 3. destination rack is not occupied by another tube
def check_pick_and_place(env: LabEnv, plan: TaskPlan) -> None:
    if plan.object not in env.objects:
        raise SafetyViolation(
            reason="unknown_object",
            message=f"Unknown object '{plan.object}'",
        )
    if plan.destination not in env.racks and plan.destination != "staging":
        raise SafetyViolation(
            reason="unknown_destination",
            message=f"Unknown destination '{plan.destination}'",
        )
    if env.state.held_object is not None:
        raise SafetyViolation(
            reason="already_holding",
            message=f"Gripper already holding '{env.state.held_object}'",
            occupant=env.state.held_object,
        )
    if env.state.gripper_open < 0.5:
        raise SafetyViolation(
            reason="gripper_not_open",
            message="Gripper must be open before starting pick-and-place",
        )

    if plan.destination in env.racks:
        blocker = occupant_at(env, plan.destination, exclude=plan.object)
        if blocker is not None:
            raise SafetyViolation(
                reason="destination_occupied",
                message=f"Destination '{plan.destination}' occupied by '{blocker}'",
                occupant=blocker,
            )
