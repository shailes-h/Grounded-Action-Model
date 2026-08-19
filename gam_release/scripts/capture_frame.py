"""Grab one frame + robot state from your adapters and save it as a sample file that
scripts/smoke_test.py understands — the fastest way to check calibration, depth, the robot
mask and the grounding BEFORE moving the robot.

    python scripts/capture_frame.py --camera my_adapters.MyCamera --robot my_adapters.MyRobot \
        --instruction "put the red cup on the tray" --out captures/frame_001.npz
    python scripts/smoke_test.py --sample captures/frame_001.npz --policy weights/gam_policy.pt \
        --wilddet weights/wilddet3d.pt --planner http://127.0.0.1:18092
    # -> captures/frame_001_smoke.png : green = planner click, blue = WildDet3D 2D box, red = 3D
    #    box projected back, yellow = robot silhouette, dimmed = cells the policy does not see
"""
import argparse
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _load(path):
    mod, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--instruction", default="pick up the object")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cam, rob = _load(a.camera)(), _load(a.robot)()
    f = cam.read()
    q = np.asarray(rob.get_state(), np.float32)
    d = dict(rgb=f.rgb, depth_m=np.asarray(f.depth_m, np.float32), K=np.asarray(f.K, np.float32),
             T_cam2base=np.asarray(f.T_cam2base, np.float32), q=q, q_prev=q, instruction=a.instruction)
    m = rob.arm_mask(f)
    if m is not None:
        d["arm_mask"] = np.asarray(m, bool)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **d)
    dv = d["depth_m"][d["depth_m"] > 0]
    print(f"saved {a.out}: rgb {f.rgb.shape}, depth valid {100 * (d['depth_m'] > 0).mean():.1f}% "
          f"range [{dv.min() if dv.size else 0:.2f}, {dv.max() if dv.size else 0:.2f}] m, "
          f"state dim {q.shape[0]}, robot mask {'yes' if m is not None else 'MISSING'}")


if __name__ == "__main__":
    main()
