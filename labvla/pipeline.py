from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from labvla.controller import ScriptedController
from labvla.env import LabEnv
from labvla.vlm import TaskPlan, build_vlm
from labvla.world_model import LightweightWorldModel


@dataclass
class PipelineResult:
    instruction: str
    plan: TaskPlan
    success: bool
    trajectory: list[dict[str, Any]]
    predicted_next_state: list[float] | None


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict[str, Any]) -> PipelineResult:
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

    instruction = str(config.get("instruction", "Move the red test tube to rack B"))
    vlm = build_vlm(str(config.get("vlm_backend", "mock")))
    plan = vlm.plan(instruction, obs.image)

    controller = ScriptedController()
    result = controller.execute(env, plan)

    object_order = ["red_tube", "blue_tube"]
    trajectory: list[dict[str, Any]] = []
    for i, action in enumerate(result.actions):
        state_vec = result.observations[i].state.to_vector(object_order)
        next_state_vec = result.observations[min(i + 1, len(result.observations) - 1)].state.to_vector(
            object_order
        )
        trajectory.append(
            {
                "state": state_vec.tolist(),
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "next_state": next_state_vec.tolist(),
            }
        )

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
        instruction=instruction,
        plan=plan,
        success=result.success,
        trajectory=trajectory,
        predicted_next_state=predicted,
    )
