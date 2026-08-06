from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class LiveViewer:
    """Mac-friendly live window that updates as simulation frames arrive."""

    def __init__(self, title: str = "LabVLA Robotics Simulation", fps: float = 12.0) -> None:
        import matplotlib

        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt

        self._plt = plt
        self._delay = 1.0 / max(fps, 1.0)
        self._fig, self._ax = plt.subplots(figsize=(9.6, 6.0))
        self._fig.canvas.manager.set_window_title(title)
        self._ax.set_axis_off()
        self._fig.tight_layout(pad=0)
        self._im: Any = None
        plt.ion()
        plt.show(block=False)

    def show(self, frame: np.ndarray) -> None:
        rgb = np.asarray(frame, dtype=np.uint8)
        if self._im is None:
            self._im = self._ax.imshow(rgb, interpolation="nearest")
        else:
            self._im.set_data(rgb)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(self._delay)

    def close(self, block: bool = True) -> None:
        self._plt.ioff()
        if block:
            self._plt.show(block=True)
        else:
            self._plt.close(self._fig)


def play_frames(frames: list[np.ndarray], fps: float = 12.0, title: str = "LabVLA Robotics Simulation") -> None:
    if not frames:
        return
    viewer = LiveViewer(title=title, fps=fps)
    for frame in frames:
        viewer.show(frame)
    viewer.close()


def make_frame_callback(viewer: LiveViewer) -> Callable[[np.ndarray], None]:
    return viewer.show
