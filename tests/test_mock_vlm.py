from __future__ import annotations

import numpy as np
import pytest

from labvla.vlm.mock_vlm import MockVLM


@pytest.fixture
def vlm() -> MockVLM:
    return MockVLM()


@pytest.fixture
def blank_image() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    ("instruction", "object_name", "destination"),
    [
        ("Move the red test tube to rack B", "red_tube", "rack_b"),
        ("Move the blue test tube to rack A", "blue_tube", "rack_a"),
        ("Move the red test tube to rack A", "red_tube", "rack_a"),
        ("put blue on rack b", "blue_tube", "rack_b"),
    ],
)
def test_mock_vlm_parses_color_and_rack(
    vlm: MockVLM,
    blank_image: np.ndarray,
    instruction: str,
    object_name: str,
    destination: str,
) -> None:
    plan = vlm.plan(instruction, blank_image)
    assert plan.object == object_name
    assert plan.destination == destination
    assert plan.action == "pick_and_place"
