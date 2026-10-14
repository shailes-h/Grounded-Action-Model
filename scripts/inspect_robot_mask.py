#!/usr/bin/env python3
"""Interactive check of the robot silhouette mask precompute.py will bake into the cache —
same URDF/extrinsics/joint-mapping call as precompute.py's `build_arm_mask_fn`, just
rendered translucent over N evenly spaced frames instead of written to an .npz.

    python scripts/inspect_robot_mask.py --config configs/pnp_v5.yaml --episode episode_0000000
    -> writes robot_mask_check.html; open it in a browser, drag the slider

This does NOT run WildDet3D / touch the cache — it's for checking `perception.robot`
(URDF, T_base_root, joint_names/affine/mirror) against your actual calibration before
burning GPU time on a full precompute.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib
for _backend in ("QtAgg", "Qt5Agg", "TkAgg", "GTK4Agg"):
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        continue
else:
    raise RuntimeError("no interactive matplotlib backend available -- "
                        "pip install pyqt5 (or apt install python3-tk)")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "precompute"))
from common.config import load_config          # noqa: E402
from common.episodes import list_episodes, load_episode  # noqa: E402
from precompute import build_arm_mask_fn        # noqa: E402  (exact function precompute.py uses)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--episode", required=True, help="episode name (hdf5 stem)")
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.45, help="mask opacity")
    ap.add_argument("--color", default="255,0,0", help="B,G,R mask color")
    args = ap.parse_args()
    color = tuple(int(c) for c in args.color.split(","))  # B,G,R

    cfg = load_config(args.config)
    arm_fn = build_arm_mask_fn(cfg)          # None if perception.robot is unset
    if arm_fn is None:
        print("[inspect] no perception.robot in config -- nothing to check")
        return

    path = next(p for p in list_episodes(cfg)
                if os.path.splitext(os.path.basename(p))[0] == args.episode)
    ep = load_episode(path, cfg)
    idxs = np.linspace(0, ep.T - 1, args.n_frames, dtype=int)

    blended_rgb = []
    for t in idxs:
        rgb = ep.rgb(int(t))
        H, W = rgb.shape[:2]
        K, E = ep.K[t].astype(np.float64), ep.E[t].astype(np.float64)
        mask = arm_fn(ep.state[t], K, E, H, W)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = bgr.copy()
        overlay[mask.astype(bool)] = color
        blended = cv2.addWeighted(overlay, args.alpha, bgr, 1 - args.alpha, 0)
        blended_rgb.append(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        print(f"[inspect] frame {t}: {int(mask.sum())} mask px")
    ep.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.15)
    im = ax.imshow(blended_rgb[0])
    ax.set_title(f"{args.episode} — frame {idxs[0]}")
    ax.axis("off")
    ax_s = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_s, "sample", 0, len(idxs) - 1, valinit=0, valstep=1)

    def update(val):
        i = int(slider.val)
        im.set_data(blended_rgb[i])
        ax.set_title(f"{args.episode} — frame {idxs[i]}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()
