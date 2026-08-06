from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labvla.pipeline import load_config, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LabVLA Robotics Simulation demo pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "default.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_pipeline(config)

    payload = {
        "instruction": result.instruction,
        "plan": result.plan.to_dict(),
        "success": result.success,
        "num_steps": len(result.trajectory),
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


if __name__ == "__main__":
    main()
