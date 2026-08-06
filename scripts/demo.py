from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labvla.pipeline import load_config, run_pipeline
from labvla.viz import save_gif
from labvla.viz.live import LiveViewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LabVLA Robotics Simulation demo pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument(
        "--gif",
        nargs="?",
        const=str(ROOT / "assets" / "demo.gif"),
        default=None,
        help="Also export a GIF (default path: assets/demo.gif)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Live playback frame rate (default: 12)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    viewer = LiveViewer(fps=args.fps)
    print("Live window open — watching pick-and-place...")
    result = run_pipeline(config, on_frame=viewer.show)

    payload = {
        "instruction": result.instruction,
        "plan": result.plan.to_dict(),
        "success": result.success,
        "num_steps": len(result.trajectory),
        "num_frames": len(result.frames),
        "predicted_next_state": result.predicted_next_state,
    }
    print(json.dumps(payload, indent=2))

    demo_cfg = config.get("demo", {})
    if demo_cfg.get("save_trajectory", True):
        out_dir = ROOT / str(demo_cfg.get("output_dir", "outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "trajectory.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "instruction": result.instruction,
                    "plan": result.plan.to_dict(),
                    "success": result.success,
                    "trajectory": result.trajectory,
                    "predicted_next_state": result.predicted_next_state,
                },
                f,
                indent=2,
            )
        print(f"saved: {out_path}")

    if args.gif and result.frames:
        gif_path = save_gif(result.frames, args.gif, duration_ms=70)
        print(f"saved: {gif_path}")

    print("Close the window to exit.")
    viewer.close()


if __name__ == "__main__":
    main()
