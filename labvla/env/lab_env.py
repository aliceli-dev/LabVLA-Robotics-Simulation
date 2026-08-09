from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class LabState:
    object_positions: dict[str, np.ndarray]
    gripper_open: float
    ee_position: np.ndarray
    success: bool = False
    held_object: str | None = None

    def to_vector(self, object_order: list[str] | None = None) -> np.ndarray:
        order = object_order or sorted(self.object_positions.keys())
        parts: list[np.ndarray] = []
        for name in order:
            pos = self.object_positions.get(name, np.zeros(3, dtype=np.float32))
            parts.append(np.asarray(pos[:2], dtype=np.float32))
        parts.append(np.asarray(self.ee_position[:2], dtype=np.float32))
        parts.append(np.array([self.gripper_open, float(self.success)], dtype=np.float32))
        return np.concatenate(parts, axis=0)


@dataclass
class LabObservation:
    image: np.ndarray
    state: LabState
    info: dict[str, Any] = field(default_factory=dict)


class LabEnv:
    def __init__(
        self,
        robot: str = "panda",
        objects: list[str] | None = None,
        racks: list[str] | None = None,
        seed: int = 0,
    ) -> None:
        self.robot = robot
        self.objects = objects or ["red_tube", "blue_tube"]
        self.racks = racks or ["rack_a", "rack_b"]
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._state = self._initial_state()
        self._base = np.array([-0.05, 0.0, 0.0], dtype=np.float32)

    def _initial_state(self) -> LabState:
        positions = {
            "red_tube": np.array([0.18, 0.12, 0.02], dtype=np.float32),
            "blue_tube": np.array([0.18, -0.12, 0.02], dtype=np.float32),
            "rack_a": np.array([-0.22, 0.12, 0.00], dtype=np.float32),
            "rack_b": np.array([-0.22, -0.12, 0.00], dtype=np.float32),
        }
        return LabState(
            object_positions=positions,
            gripper_open=1.0,
            ee_position=np.array([0.05, 0.0, 0.22], dtype=np.float32),
            success=False,
            held_object=None,
        )

    def reset(self) -> LabObservation:
        self._rng = np.random.default_rng(self.seed)
        self._state = self._initial_state()
        return self._observe()

    def world_to_pixel(self, pos: np.ndarray, width: int, height: int) -> tuple[int, int]:
        u = int((pos[0] + 0.38) / 0.76 * (width - 1))
        v = int((0.32 - pos[1]) / 0.64 * (height - 1))
        return int(np.clip(u, 0, width - 1)), int(np.clip(v, 0, height - 1))

    def render_rgb(
        self,
        width: int = 640,
        height: int = 400,
        instruction: str | None = None,
        plan_text: str | None = None,
        status: str | None = None,
    ) -> np.ndarray:
        img = Image.new("RGB", (width, height), (246, 244, 239))
        draw = ImageDraw.Draw(img)
        sx = width / 640.0
        sy = height / 400.0
        s = min(sx, sy)
        font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
        try:
            font = ImageFont.truetype(font_path, max(8, int(16 * s)))
            font_sm = ImageFont.truetype(font_path, max(7, int(13 * s)))
            font_lg = ImageFont.truetype(font_path, max(9, int(18 * s)))
        except OSError:
            font = ImageFont.load_default()
            font_sm = font
            font_lg = font

        margin = max(4, int(24 * sx))
        header_h = max(18, int(58 * sy))
        footer = max(8, int(28 * sy))
        bench_top = header_h + max(4, int(12 * sy))
        bench_bottom = height - footer
        if bench_bottom <= bench_top + 4:
            bench_top = header_h + 2
            bench_bottom = height - 4
        inset = max(2, int(18 * sx))
        pad_top = max(2, int(18 * sy))
        pad_bottom = max(2, int(18 * sy))
        inner_top = bench_top + pad_top
        inner_bottom = bench_bottom - pad_bottom
        if inner_bottom < inner_top:
            inner_top = bench_top + 1
            inner_bottom = bench_bottom - 1

        draw.rounded_rectangle(
            [margin, bench_top, width - margin, bench_bottom],
            radius=max(2, int(18 * s)),
            fill=(214, 201, 178),
            outline=(120, 98, 72),
            width=max(1, int(2 * s)),
        )
        if width - margin - inset > margin + inset and inner_bottom >= inner_top:
            draw.rectangle(
                [margin + inset, inner_top, width - margin - inset, inner_bottom],
                fill=(232, 222, 205),
            )

        for rack_name, label in (("rack_a", "Rack A"), ("rack_b", "Rack B")):
            pos = self._state.object_positions[rack_name]
            x, y = self.world_to_pixel(pos, width, height)
            rw, rh = max(6, int(34 * s)), max(5, int(28 * s))
            draw.rounded_rectangle(
                [x - rw, y - rh, x + rw, y + rh],
                radius=max(2, int(8 * s)),
                fill=(90, 96, 110),
                outline=(40, 44, 54),
                width=max(1, int(2 * s)),
            )
            for slot_dx in (-int(16 * s), 0, int(16 * s)):
                r = max(2, int(7 * s))
                draw.ellipse(
                    [x + slot_dx - r, y - int(8 * s), x + slot_dx + r, y + int(10 * s)],
                    outline=(200, 205, 215),
                    width=max(1, int(2 * s)),
                )
            if height >= 160:
                draw.text((x - int(22 * s), y + rh + 2), label, fill=(55, 55, 60), font=font_sm)

        colors = {
            "red_tube": ((198, 48, 48), (255, 170, 170)),
            "blue_tube": ((42, 92, 196), (170, 200, 255)),
        }
        for name, (fill, rim) in colors.items():
            pos = self._state.object_positions[name]
            x, y = self.world_to_pixel(pos, width, height)
            tw, th = max(3, int(8 * s)), max(5, int(18 * s))
            draw.rounded_rectangle(
                [x - tw, y - th, x + tw, y + th],
                radius=max(2, int(6 * s)),
                fill=fill,
                outline=rim,
                width=max(1, int(2 * s)),
            )
            draw.ellipse(
                [x - tw, y - th - max(2, int(4 * s)), x + tw, y - th + max(2, int(6 * s))],
                fill=rim,
                outline=fill,
                width=1,
            )

        base_x, base_y = self.world_to_pixel(self._base, width, height)
        ee_x, ee_y = self.world_to_pixel(self._state.ee_position, width, height)
        mid_x = int(0.55 * base_x + 0.45 * ee_x)
        mid_y = int(0.35 * base_y + 0.65 * ee_y) - int(36 * sy)
        br = max(4, int(16 * s))
        draw.ellipse(
            [base_x - br, base_y - int(12 * s), base_x + br, base_y + int(12 * s)],
            fill=(55, 58, 66),
            outline=(25, 25, 30),
            width=max(1, int(2 * s)),
        )
        draw.line(
            [(base_x, base_y), (mid_x, mid_y), (ee_x, ee_y)],
            fill=(70, 74, 84),
            width=max(2, int(8 * s)),
        )
        jr = max(2, int(8 * s))
        draw.ellipse([mid_x - jr, mid_y - jr, mid_x + jr, mid_y + jr], fill=(95, 100, 112))
        gap = max(2, int((10 if self._state.gripper_open > 0.5 else 3) * s))
        gw = max(2, int(14 * s))
        draw.rectangle([ee_x - gw - gap, ee_y - int(4 * s), ee_x - gap, ee_y + int(10 * s)], fill=(40, 40, 45))
        draw.rectangle([ee_x + gap, ee_y - int(4 * s), ee_x + gw + gap, ee_y + int(10 * s)], fill=(40, 40, 45))
        er = max(2, int(7 * s))
        draw.ellipse(
            [ee_x - er, ee_y - er, ee_x + er, ee_y + er],
            fill=(230, 180, 60),
            outline=(120, 80, 20),
            width=max(1, int(2 * s)),
        )

        draw.rectangle([0, 0, width, header_h], fill=(28, 36, 48))
        if width >= 200:
            draw.text((max(4, int(16 * sx)), max(2, int(10 * sy))), "LabVLA Robotics Simulation", fill=(245, 245, 245), font=font_lg)
            label = status or ("SUCCESS" if self._state.success else "RUNNING")
            status_colors = {
                "SUCCESS": (96, 210, 140),
                "RUNNING": (240, 190, 90),
                "FAILED": (230, 90, 90),
                "RETRY": (255, 170, 70),
            }
            status_color = status_colors.get(label, (240, 190, 90))
            draw.text((width - max(50, int(110 * sx)), max(2, int(14 * sy))), label, fill=status_color, font=font)
            if instruction:
                draw.text((max(4, int(16 * sx)), max(10, int(34 * sy))), f"Instruction: {instruction}", fill=(190, 205, 220), font=font_sm)
        if plan_text and height >= 160:
            draw.text((max(4, int(16 * sx)), height - max(10, int(22 * sy))), plan_text, fill=(70, 70, 75), font=font_sm)

        return np.asarray(img, dtype=np.uint8)

    def _observe(self) -> LabObservation:
        image = self.render_rgb(width=128, height=128)
        return LabObservation(image=image, state=self._state, info={"robot": self.robot})

    def set_ee(self, position: np.ndarray, gripper_open: float | None = None) -> LabObservation:
        self._state.ee_position = np.asarray(position, dtype=np.float32).copy()
        if gripper_open is not None:
            self._state.gripper_open = float(np.clip(gripper_open, 0.0, 1.0))
        held = self._state.held_object
        if held is not None:
            obj = self._state.ee_position.copy()
            obj[2] = 0.02
            self._state.object_positions[held] = obj
        return self._observe()

    def grasp(self, object_name: str) -> LabObservation:
        self._state.held_object = object_name
        self._state.gripper_open = 0.0
        return self.set_ee(self._state.ee_position, gripper_open=0.0)

    def release(self) -> LabObservation:
        self._state.held_object = None
        self._state.gripper_open = 1.0
        return self._observe()

    def step(self, action: np.ndarray) -> LabObservation:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        delta = action[:3] if action.shape[0] >= 3 else np.zeros(3, dtype=np.float32)
        grip = float(action[3]) if action.shape[0] >= 4 else self._state.gripper_open
        return self.set_ee(self._state.ee_position + delta, gripper_open=grip)

    def apply_placement(self, object_name: str, destination: str) -> LabObservation:
        if object_name not in self._state.object_positions:
            raise KeyError(object_name)
        if destination not in self._state.object_positions:
            raise KeyError(destination)
        target = self._state.object_positions[destination].copy()
        target[2] = 0.02
        self._state.object_positions[object_name] = target
        self._state.ee_position = target + np.array([0.0, 0.0, 0.12], dtype=np.float32)
        self._state.gripper_open = 1.0
        self._state.held_object = None
        self._state.success = True
        return self._observe()

    def apply_miss(
        self,
        object_name: str,
        destination: str,
        offset: tuple[float, float, float] = (0.07, 0.05, 0.02),
    ) -> LabObservation:
        """Drop the object near the destination (failed placement)."""
        if object_name not in self._state.object_positions:
            raise KeyError(object_name)
        if destination not in self._state.object_positions:
            raise KeyError(destination)
        target = self._state.object_positions[destination].copy()
        target = target + np.asarray(offset, dtype=np.float32)
        target[2] = 0.02
        self._state.object_positions[object_name] = target
        self._state.ee_position = target + np.array([0.0, 0.0, 0.12], dtype=np.float32)
        self._state.gripper_open = 1.0
        self._state.held_object = None
        self._state.success = False
        return self._observe()

    @property
    def state(self) -> LabState:
        return self._state
