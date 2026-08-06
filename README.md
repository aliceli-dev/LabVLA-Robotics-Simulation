# LabVLA Robotics Simulation

This is a Python project for language-guided laboratory robotics in simulation.

The goal is simple: you give a natural-language instruction like **"Move the red test tube to rack B"**, the system looks at the simulated lab table through a camera, figures out what to do, and a virtual Franka Panda arm carries out the pick-and-place.

## What it does

1. Takes a language instruction and a camera image from the simulator
2. Uses a Vision-Language Model (VLM) to turn that into a structured task plan
3. Runs the plan with a robotic controller in simulation
4. Logs states and actions, then uses a lightweight world model to predict what happens next

This is meant as a practical Mac-friendly demo of the VLM → robot control → world model pipeline used in lab automation research. It is **not** claiming to be a full end-to-end production VLA yet — the VLM plans, and a controller executes. A true VLA action head is left as a later extension.

## Tech stack

| Piece | Choice |
|---|---|
| Language | Python |
| Physics / robot sim | MuJoCo + robosuite |
| Robot | Franka Panda |
| Perception + language | Pluggable VLM (mock / API / local) |
| Control | Scripted / OSC / IK pick-and-place |
| World model | Lightweight PyTorch model over low-dimensional state |

Built to run on a Mac without an NVIDIA GPU.

## Status

Early stage — repository setup and README first. Simulation environment, VLM interface, controller, and world model will be added next.
