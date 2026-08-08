from __future__ import annotations

from collections.abc import Callable

import numpy as np

from labvla.env import LabEnv
from labvla.vlm import TaskPlan

from .base import ControlResult, Controller

FrameCallback = Callable[[np.ndarray], None]


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


class ScriptedController(Controller):
    def __init__(
        self,
        steps_per_segment: int = 8,
        on_frame: FrameCallback | None = None,
        render_size: tuple[int, int] = (720, 450),
    ) -> None:
        self.steps_per_segment = steps_per_segment
        self.on_frame = on_frame
        self.render_size = render_size

    def _render(self, env: LabEnv, instruction: str, plan: TaskPlan) -> np.ndarray:
        width, height = self.render_size
        return env.render_rgb(
            width=width,
            height=height,
            instruction=instruction,
            plan_text=f'VLM plan: {{"object": "{plan.object}", "destination": "{plan.destination}"}}',
        )

    def _push_frame(
        self,
        frames: list[np.ndarray],
        env: LabEnv,
        instruction: str,
        plan: TaskPlan,
    ) -> None:
        frame = self._render(env, instruction, plan)
        frames.append(frame)
        if self.on_frame is not None:
            self.on_frame(frame)

    def _move_to(
        self,
        env: LabEnv,
        target: np.ndarray,
        gripper_open: float,
        actions: list[np.ndarray],
        observations: list,
        frames: list[np.ndarray],
        instruction: str,
        plan: TaskPlan,
    ) -> None:
        start = env.state.ee_position.copy()
        for i in range(1, self.steps_per_segment + 1):
            t = i / self.steps_per_segment
            pose = _lerp(start, target, t)
            delta = pose - env.state.ee_position
            action = np.array([*delta, gripper_open], dtype=np.float32)
            obs = env.set_ee(pose, gripper_open=gripper_open)
            actions.append(action)
            observations.append(obs)
            self._push_frame(frames, env, instruction, plan)

    def execute(self, env: LabEnv, plan: TaskPlan, instruction: str = "") -> ControlResult:
        env.state.success = False
        obs0 = env._observe()
        object_pos = env.state.object_positions[plan.object].copy()
        dest_pos = env.state.object_positions[plan.destination].copy()

        above_obj = object_pos + np.array([0.0, 0.0, 0.14], dtype=np.float32)
        grasp_pose = object_pos + np.array([0.0, 0.0, 0.04], dtype=np.float32)
        lift_pose = object_pos + np.array([0.0, 0.0, 0.16], dtype=np.float32)
        above_dest = dest_pos + np.array([0.0, 0.0, 0.16], dtype=np.float32)
        place_pose = dest_pos + np.array([0.0, 0.0, 0.04], dtype=np.float32)
        retreat = dest_pos + np.array([0.0, 0.0, 0.14], dtype=np.float32)

        actions: list[np.ndarray] = []
        observations = [obs0]
        frames: list[np.ndarray] = []
        self._push_frame(frames, env, instruction, plan)

        self._move_to(env, above_obj, 1.0, actions, observations, frames, instruction, plan)
        self._move_to(env, grasp_pose, 1.0, actions, observations, frames, instruction, plan)
        env.grasp(plan.object)
        self._push_frame(frames, env, instruction, plan)
        self._move_to(env, lift_pose, 0.0, actions, observations, frames, instruction, plan)
        self._move_to(env, above_dest, 0.0, actions, observations, frames, instruction, plan)
        self._move_to(env, place_pose, 0.0, actions, observations, frames, instruction, plan)
        final_obs = env.apply_placement(plan.object, plan.destination)
        observations.append(final_obs)
        self._push_frame(frames, env, instruction, plan)
        self._move_to(env, retreat, 1.0, actions, observations, frames, instruction, plan)
        for _ in range(6):
            frame = frames[-1]
            frames.append(frame)
            if self.on_frame is not None:
                self.on_frame(frame)

        return ControlResult(
            success=final_obs.state.success,
            actions=actions,
            observations=observations,
            info={
                "object": plan.object,
                "destination": plan.destination,
                "frames": frames,
            },
        )
