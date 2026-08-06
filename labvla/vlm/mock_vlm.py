from __future__ import annotations

import re

import numpy as np

from .base import TaskPlan, VLMBackend


class MockVLM(VLMBackend):
    def plan(self, instruction: str, image: np.ndarray) -> TaskPlan:
        _ = image
        text = instruction.lower()
        color = "red"
        if "blue" in text:
            color = "blue"
        object_name = f"{color}_tube"
        destination = "rack_b"
        if re.search(r"rack\s*a", text):
            destination = "rack_a"
        elif re.search(r"rack\s*b", text):
            destination = "rack_b"
        return TaskPlan(object=object_name, destination=destination)
