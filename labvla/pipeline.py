from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from labvla.controller import ControlExecutionError, ScriptedController
from labvla.env import LabEnv
from labvla.safety import SafetyViolation, check_pick_and_place
from labvla.vlm import TaskPlan, build_vlm
from labvla.world_model import LightweightWorldModel

FrameCallback = Callable[[np.ndarray], None]


@dataclass
class AttemptRecord:
    instruction: str
    attempt: int
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "attempt": self.attempt,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class SafetyEvent:
    instruction: str
    reason: str
    message: str
    occupant: str | None
    action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "reason": self.reason,
            "message": self.message,
            "occupant": self.occupant,
            "action": self.action,
        }


@dataclass
class PipelineResult:
    instruction: str
    plan: TaskPlan
    instructions: list[str]
    plans: list[TaskPlan]
    success: bool
    trajectory: list[dict[str, Any]]
    predicted_next_state: list[float] | None
    frames: list[np.ndarray]
    attempts: list[AttemptRecord] = field(default_factory=list)
    safety_events: list[SafetyEvent] = field(default_factory=list)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_instructions(config: dict[str, Any]) -> list[str]:
    raw = config.get("instructions")
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw]
    return [str(config.get("instruction", "Move the red test tube to rack B"))]


def _append_trajectory(
    trajectory: list[dict[str, Any]],
    actions: list[np.ndarray],
    observations: list,
    instruction: str,
    attempt: int,
    object_order: list[str],
) -> None:
    for i, action in enumerate(actions):
        idx = min(i, len(observations) - 1)
        next_idx = min(i + 1, len(observations) - 1)
        state_vec = observations[idx].state.to_vector(object_order)
        next_state_vec = observations[next_idx].state.to_vector(object_order)
        trajectory.append(
            {
                "state": state_vec.tolist(),
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "next_state": next_state_vec.tolist(),
                "instruction": instruction,
                "attempt": attempt,
            }
        )


def _push_banner_frames(
    env: LabEnv,
    frames: list[np.ndarray],
    *,
    instruction: str,
    plan: TaskPlan,
    status: str,
    on_frame: FrameCallback | None,
    n: int = 8,
) -> None:
    for _ in range(n):
        frame = env.render_rgb(
            width=720,
            height=450,
            instruction=instruction,
            plan_text=f'VLM plan: {{"object": "{plan.object}", "destination": "{plan.destination}"}}',
            status=status,
        )
        frames.append(frame)
        if on_frame is not None:
            on_frame(frame)


def _ensure_safe_to_execute(
    env: LabEnv,
    plan: TaskPlan,
    instruction: str,
    controller: ScriptedController,
    *,
    auto_clear: bool,
    frames: list[np.ndarray],
    trajectory: list[dict[str, Any]],
    attempts: list[AttemptRecord],
    safety_events: list[SafetyEvent],
    object_order: list[str],
    on_frame: FrameCallback | None,
) -> bool:
    try:
        check_pick_and_place(env, plan)
        return True
    except SafetyViolation as violation:
        safety_events.append(
            SafetyEvent(
                instruction=instruction,
                reason=violation.reason,
                message=violation.message,
                occupant=violation.occupant,
                action="blocked",
            )
        )
        attempts.append(
            AttemptRecord(
                instruction=instruction,
                attempt=0,
                success=False,
                error=f"BLOCKED: {violation.message}",
            )
        )
        _push_banner_frames(
            env,
            frames,
            instruction=f"BLOCKED: {violation.message}",
            plan=plan,
            status="BLOCKED",
            on_frame=on_frame,
        )

        if not auto_clear or violation.reason != "destination_occupied" or not violation.occupant:
            return False

        clear_plan = TaskPlan(object=violation.occupant, destination="staging")
        clear_instruction = f"Safety clear: move {violation.occupant} to staging"
        clear_result = controller.execute(
            env,
            clear_plan,
            instruction=clear_instruction,
            force_fail=False,
            attempt=1,
        )
        frames.extend(clear_result.info.get("frames", []))
        _append_trajectory(
            trajectory,
            clear_result.actions,
            clear_result.observations,
            clear_instruction,
            1,
            object_order,
        )
        attempts.append(
            AttemptRecord(
                instruction=clear_instruction,
                attempt=1,
                success=bool(clear_result.success),
                error=None,
            )
        )
        safety_events.append(
            SafetyEvent(
                instruction=instruction,
                reason=violation.reason,
                message=f"Cleared '{violation.occupant}' to staging",
                occupant=violation.occupant,
                action="cleared",
            )
        )

        try:
            check_pick_and_place(env, plan)
        except SafetyViolation:
            return False
        return True


def run_pipeline(
    config: dict[str, Any],
    on_frame: FrameCallback | None = None,
) -> PipelineResult:
    env_cfg = config.get("env", {})
    wm_cfg = config.get("world_model", {})
    demo_cfg = config.get("demo", {})

    env = LabEnv(
        robot=env_cfg.get("robot", "panda"),
        objects=env_cfg.get("objects"),
        racks=env_cfg.get("racks"),
        seed=int(demo_cfg.get("seed", 0)),
    )
    obs = env.reset()

    instructions = _load_instructions(config)
    vlm = build_vlm(str(config.get("vlm_backend", "mock")))
    controller = ScriptedController(
        steps_per_segment=int(demo_cfg.get("steps_per_segment", 8)),
        on_frame=on_frame,
    )

    max_retries = max(0, int(demo_cfg.get("max_retries", 1)))
    simulate_first_failure = bool(demo_cfg.get("simulate_first_failure", False))
    safety_check = bool(demo_cfg.get("safety_check", True))
    auto_clear_blocked = bool(demo_cfg.get("auto_clear_blocked", True))

    plans: list[TaskPlan] = []
    frames: list[np.ndarray] = []
    object_order = ["red_tube", "blue_tube"]
    trajectory: list[dict[str, Any]] = []
    attempts: list[AttemptRecord] = []
    safety_events: list[SafetyEvent] = []
    all_success = True
    simulated_failure_used = False

    for instruction in instructions:
        plan = vlm.plan(instruction, obs.image)
        plans.append(plan)
        task_success = False

        if safety_check:
            ok = _ensure_safe_to_execute(
                env,
                plan,
                instruction,
                controller,
                auto_clear=auto_clear_blocked,
                frames=frames,
                trajectory=trajectory,
                attempts=attempts,
                safety_events=safety_events,
                object_order=object_order,
                on_frame=on_frame,
            )
            if not ok:
                all_success = False
                continue
            if env.state.object_positions:
                obs = env._observe()

        for attempt in range(1, max_retries + 2):
            force_fail = (
                simulate_first_failure
                and not simulated_failure_used
                and attempt == 1
                and instruction == instructions[0]
            )
            try:
                result = controller.execute(
                    env,
                    plan,
                    instruction=instruction,
                    force_fail=force_fail,
                    attempt=attempt,
                )
            except ControlExecutionError as exc:
                if force_fail:
                    simulated_failure_used = True
                frames.extend(exc.frames)
                _append_trajectory(
                    trajectory,
                    exc.actions,
                    exc.observations,
                    instruction,
                    attempt,
                    object_order,
                )
                if exc.observations:
                    obs = exc.observations[-1]
                attempts.append(
                    AttemptRecord(
                        instruction=instruction,
                        attempt=attempt,
                        success=False,
                        error=str(exc),
                    )
                )
                if attempt > max_retries:
                    break
                retry_label = f"{instruction}  (retry {attempt + 1}/{max_retries + 1})"
                _push_banner_frames(
                    env,
                    frames,
                    instruction=retry_label,
                    plan=plan,
                    status="RETRY",
                    on_frame=on_frame,
                    n=6,
                )
                continue

            frames.extend(result.info.get("frames", []))
            _append_trajectory(
                trajectory,
                result.actions,
                result.observations,
                instruction,
                attempt,
                object_order,
            )
            if result.observations:
                obs = result.observations[-1]
            attempts.append(
                AttemptRecord(
                    instruction=instruction,
                    attempt=attempt,
                    success=bool(result.success),
                    error=None,
                )
            )
            task_success = bool(result.success)
            break

        all_success = all_success and task_success

    predicted: list[float] | None = None
    if trajectory:
        model = LightweightWorldModel(
            state_dim=int(wm_cfg.get("state_dim", 8)),
            action_dim=int(wm_cfg.get("action_dim", 4)),
            hidden_dim=int(wm_cfg.get("hidden_dim", 64)),
        )
        pred = model.predict_next(
            np.asarray(trajectory[0]["state"], dtype=np.float32),
            np.asarray(trajectory[0]["action"], dtype=np.float32),
        )
        predicted = pred.reshape(-1).tolist()

    return PipelineResult(
        instruction=" → ".join(instructions),
        plan=plans[-1],
        instructions=instructions,
        plans=plans,
        success=all_success,
        trajectory=trajectory,
        predicted_next_state=predicted,
        frames=frames,
        attempts=attempts,
        safety_events=safety_events,
    )
