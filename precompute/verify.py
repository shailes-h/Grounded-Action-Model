"""Visual check of one cached episode BEFORE training: overlay clicks, 2D boxes, projected
3D boxes and the patch mask on the raw video.

    python precompute/verify.py --config configs/my_task.yaml --episode episode_0000000 [--out check.mp4]

Look for: boxes tight on the objects, 3D box projections landing on them (calibration),
the robot dimly visible inside the kept patch area.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.config import load_config  # noqa: E402
from common.episodes import list_episodes, load_episode  # noqa: E402
from gam.perception.geometry import box2d_from_obb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--episode", required=True, help="episode name (npz stem)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    z = np.load(os.path.join(cfg["train"]["cache_dir"], args.episode + ".npz"), allow_pickle=True)
    src = [p for p in list_episodes(cfg) if args.episode in p]
    assert src, f"raw episode {args.episode} not found"
    ep = load_episode(src[0], cfg)
    out = args.out or os.path.join(cfg["train"]["cache_dir"], args.episode + "_check.mp4")
    H, W = ep.rgb(0).shape[:2]
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
    for t in range(ep.T):
        vis = cv2.cvtColor(ep.rgb(t), cv2.COLOR_RGB2BGR).copy()
        G = int(round(z["patches"].shape[1] ** 0.5))
        m16 = (np.abs(z["patches"][t]).sum(-1) > 0).reshape(G, G)
        big = cv2.resize(m16.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
        vis = np.where(big[..., None] > 0, vis, (vis * 0.35).astype(np.uint8))
        for s in range(z["box2d"].shape[1]):
            b = z["box2d"][t, s]
            if b[2] > b[0]:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 160, 0), 2)
            if z["boxgeo"][t, s, :3].any():
                R = np.eye(3); R[:, 0] = z["boxgeo"][t, s, 3:6]; R[:, 1] = z["boxgeo"][t, s, 6:9]
                R[:, 2] = np.cross(R[:, 0], R[:, 1])
                b3 = box2d_from_obb(z["xyz"][t, s], z["boxgeo"][t, s, :3], R, ep.E[t], ep.K[t])
                if b3.any():
                    cv2.rectangle(vis, (int(b3[0]), int(b3[1])), (int(b3[2]), int(b3[3])), (0, 0, 255), 2)
            c = z["clicks"][t, s]
            if np.isfinite(c).all():
                cv2.drawMarker(vis, (int(c[0]), int(c[1])), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
        vw.write(vis)
        if t % 50 == 0:
            print(f"frame {t}/{ep.T}", flush=True)
    vw.release()
    ep.close()
    print("wrote", out, "\nVERIFY_DONE")


if __name__ == "__main__":
    main()
