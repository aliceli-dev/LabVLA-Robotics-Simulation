from __future__ import annotations

from collections.abc import Callable

import numpy as np

from labvla.env import LabEnv
from labvla.vlm import TaskPlan

from .base import ControlResult, Controller
from .errors import ControlExecutionError

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

    def _render(
        self,
        env: LabEnv,
        instruction: str,
        plan: TaskPlan,
        status: str | None = None,
    ) -> np.ndarray:
        width, height = self.render_size
        return env.render_rgb(
            width=width,
            height=height,
            instruction=instruction,
            plan_text=f'VLM plan: {{"object": "{plan.object}", "destination": "{plan.destination}"}}',
            status=status,
        )

    def _push_frame(
        self,
        frames: list[np.ndarray],
        env: LabEnv,
        instruction: str,
        plan: TaskPlan,
        status: str | None = None,
    ) -> None:
        frame = self._render(env, instruction, plan, status=status)
        frames.append(frame)
        if self.on_frame is not None:
            self.on_frame(frame)

    def _hold(
        self,
        frames: list[np.ndarray],
        env: LabEnv,
        instruction: str,
        plan: TaskPlan,
        n: int = 6,
        status: str | None = None,
    ) -> None:
        for _ in range(n):
            self._push_frame(frames, env, instruction, plan, status=status)

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
        status: str | None = None,
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
            self._push_frame(frames, env, instruction, plan, status=status)

    def execute(
        self,
        env: LabEnv,
        plan: TaskPlan,
        instruction: str = "",
        *,
        force_fail: bool = False,
        attempt: int = 1,
    ) -> ControlResult:
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
        run_status = "RETRY" if attempt > 1 else None
        attempt_label = f"{instruction}  (attempt {attempt})"

        self._push_frame(frames, env, attempt_label, plan, status=run_status)
        self._move_to(
            env, above_obj, 1.0, actions, observations, frames, attempt_label, plan, status=run_status
        )
        self._move_to(
            env, grasp_pose, 1.0, actions, observations, frames, attempt_label, plan, status=run_status
        )
        env.grasp(plan.object)
        self._push_frame(frames, env, attempt_label, plan, status=run_status)
        self._move_to(
            env, lift_pose, 0.0, actions, observations, frames, attempt_label, plan, status=run_status
        )
        self._move_to(
            env, above_dest, 0.0, actions, observations, frames, attempt_label, plan, status=run_status
        )

        if force_fail:
            miss_pose = place_pose + np.array([0.07, 0.05, 0.0], dtype=np.float32)
            self._move_to(
                env, miss_pose, 0.0, actions, observations, frames, attempt_label, plan, status="FAILED"
            )
            final_obs = env.apply_miss(plan.object, plan.destination)
            observations.append(final_obs)
            self._push_frame(frames, env, attempt_label, plan, status="FAILED")
            self._move_to(
                env, retreat, 1.0, actions, observations, frames, attempt_label, plan, status="FAILED"
            )
            self._hold(frames, env, attempt_label, plan, n=8, status="FAILED")
            raise ControlExecutionError(
                f"Placement missed for {plan.object} → {plan.destination}",
                attempt=attempt,
                actions=actions,
                observations=observations,
                frames=frames,
                info={
                    "object": plan.object,
                    "destination": plan.destination,
                    "attempt": attempt,
                    "reason": "placement_miss",
                },
            )

        self._move_to(
            env, place_pose, 0.0, actions, observations, frames, attempt_label, plan, status=run_status
        )
        final_obs = env.apply_placement(plan.object, plan.destination)
        observations.append(final_obs)
        self._push_frame(frames, env, attempt_label, plan, status="SUCCESS")
        self._move_to(
            env, retreat, 1.0, actions, observations, frames, attempt_label, plan, status="SUCCESS"
        )
        self._hold(frames, env, attempt_label, plan, n=6, status="SUCCESS")

        return ControlResult(
            success=final_obs.state.success,
            actions=actions,
            observations=observations,
            info={
                "object": plan.object,
                "destination": plan.destination,
                "frames": frames,
                "attempt": attempt,
            },
        )
