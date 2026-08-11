from __future__ import annotations

import pytest

from labvla.env import LabEnv
from labvla.safety import SafetyViolation, check_pick_and_place, occupant_at
from labvla.vlm import TaskPlan


@pytest.fixture
def env() -> LabEnv:
    lab = LabEnv(seed=0)
    lab.reset()
    return lab


def test_occupant_at_empty_rack(env: LabEnv) -> None:
    assert occupant_at(env, "rack_a") is None
    assert occupant_at(env, "rack_b") is None


def test_occupant_at_detects_placed_tube(env: LabEnv) -> None:
    env.apply_placement("blue_tube", "rack_a")
    assert occupant_at(env, "rack_a") == "blue_tube"
    assert occupant_at(env, "rack_a", exclude="blue_tube") is None


def test_check_allows_free_destination(env: LabEnv) -> None:
    plan = TaskPlan(object="red_tube", destination="rack_b")
    check_pick_and_place(env, plan)


def test_check_blocks_occupied_destination(env: LabEnv) -> None:
    env.apply_placement("blue_tube", "rack_a")
    plan = TaskPlan(object="red_tube", destination="rack_a")
    with pytest.raises(SafetyViolation) as exc_info:
        check_pick_and_place(env, plan)
    assert exc_info.value.reason == "destination_occupied"
    assert exc_info.value.occupant == "blue_tube"


def test_check_blocks_unknown_object(env: LabEnv) -> None:
    plan = TaskPlan(object="green_tube", destination="rack_a")
    with pytest.raises(SafetyViolation) as exc_info:
        check_pick_and_place(env, plan)
    assert exc_info.value.reason == "unknown_object"


def test_check_blocks_closed_gripper(env: LabEnv) -> None:
    env.state.gripper_open = 0.0
    plan = TaskPlan(object="red_tube", destination="rack_b")
    with pytest.raises(SafetyViolation) as exc_info:
        check_pick_and_place(env, plan)
    assert exc_info.value.reason == "gripper_not_open"


def test_check_blocks_already_holding(env: LabEnv) -> None:
    env.grasp("red_tube")
    plan = TaskPlan(object="blue_tube", destination="rack_a")
    with pytest.raises(SafetyViolation) as exc_info:
        check_pick_and_place(env, plan)
    assert exc_info.value.reason == "already_holding"
    assert exc_info.value.occupant == "red_tube"


def test_safety_violation_to_dict() -> None:
    err = SafetyViolation(
        reason="destination_occupied",
        message="blocked",
        occupant="blue_tube",
    )
    assert err.to_dict() == {
        "reason": "destination_occupied",
        "message": "blocked",
        "occupant": "blue_tube",
    }
