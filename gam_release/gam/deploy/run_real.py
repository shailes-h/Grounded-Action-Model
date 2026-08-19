"""Real-robot control loop.

    python -m gam.deploy.run_real --config configs/deploy_example.yaml \
        --instruction "put the red cup on the tray" --camera my_pkg.MyCamera --robot my_pkg.MyRobot

`--camera` / `--robot` are import paths of classes implementing gam.deploy.interfaces
(Camera / Robot). Start the planner first: `python -m gam.planner.server`.
"""
from __future__ import annotations

import argparse
import importlib
import time

import numpy as np
import yaml

from .agent import AgentConfig, GAMAgent


def _load(path: str):
    mod, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="yaml with AgentConfig fields")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--camera", required=True, help="import path of a Camera implementation")
    ap.add_argument("--robot", required=True, help="import path of a Robot implementation")
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--hz", type=float, default=15.0, help="control rate (matches training data)")
    args = ap.parse_args()
    cfg = AgentConfig(**yaml.safe_load(open(args.config)))
    agent = GAMAgent(cfg)
    camera, robot = _load(args.camera)(), _load(args.robot)()
    agent.reset(args.instruction)
    dt = 1.0 / args.hz
    steps = 0
    while steps < args.max_steps:
        frame = camera.read()
        q = robot.get_state()
        chunk = agent.act(frame, q, arm_mask=robot.arm_mask(frame))
        for a in chunk:
            t0 = time.time()
            robot.apply_action(np.asarray(a, np.float32))
            agent.push_state(robot.get_state())
            steps += 1
            if steps >= args.max_steps:
                break
            time.sleep(max(0.0, dt - (time.time() - t0)))
        print(f"[run] t={agent.t} steps={steps} memory: {agent.memory_state}", flush=True)


if __name__ == "__main__":
    main()
