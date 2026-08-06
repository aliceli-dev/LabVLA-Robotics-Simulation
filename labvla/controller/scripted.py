from __future__ import annotations

import numpy as np

from labvla.env import LabEnv
from labvla.vlm import TaskPlan

from .base import ControlResult, Controller


class ScriptedController(Controller):
    def execute(self, env: LabEnv, plan: TaskPlan) -> ControlResult:
        obs0 = env._observe()
        object_pos = env.state.object_positions[plan.object]
        dest_pos = env.state.object_positions[plan.destination]

        approach = dest_pos - object_pos
        actions = [
            np.array([*(object_pos - env.state.ee_position), 0.0], dtype=np.float32),
            np.array([0.0, 0.0, -0.05, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.05, 0.0], dtype=np.float32),
            np.array([*approach, 1.0], dtype=np.float32),
        ]

        observations = [obs0]
        for action in actions[:-1]:
            observations.append(env.step(action))
        final_obs = env.apply_placement(plan.object, plan.destination)
        observations.append(final_obs)

        return ControlResult(
            success=final_obs.state.success,
            actions=actions,
            observations=observations,
            info={"object": plan.object, "destination": plan.destination},
        )
