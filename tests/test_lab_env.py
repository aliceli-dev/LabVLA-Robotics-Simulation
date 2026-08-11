from __future__ import annotations

import numpy as np

from labvla.env import LabEnv


def test_reset_restores_initial_layout() -> None:
    env = LabEnv(seed=0)
    env.reset()
    env.apply_placement("red_tube", "rack_b")
    assert env.state.success is True

    obs = env.reset()
    assert obs.state.success is False
    assert obs.state.held_object is None
    np.testing.assert_allclose(
        obs.state.object_positions["red_tube"],
        np.array([0.18, 0.12, 0.02], dtype=np.float32),
    )


def test_apply_placement_moves_object_to_rack() -> None:
    env = LabEnv(seed=0)
    env.reset()
    obs = env.apply_placement("blue_tube", "rack_a")
    rack = env.state.object_positions["rack_a"]
    tube = obs.state.object_positions["blue_tube"]
    np.testing.assert_allclose(tube[:2], rack[:2])
    np.testing.assert_allclose(tube[2], 0.02)
    assert obs.state.held_object is None
    assert obs.state.gripper_open == 1.0
    assert obs.state.success is True


def test_apply_miss_leaves_object_near_rack() -> None:
    env = LabEnv(seed=0)
    env.reset()
    obs = env.apply_miss("red_tube", "rack_b")
    rack = env.state.object_positions["rack_b"]
    tube = obs.state.object_positions["red_tube"]
    assert float(np.linalg.norm(tube[:2] - rack[:2])) > 0.05
    assert obs.state.success is False
    assert obs.state.held_object is None
