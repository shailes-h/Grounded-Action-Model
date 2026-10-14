#!/usr/bin/env python3
"""Local search around the current perception.robot params (left/right arm distance from
origin, global x/y/z/roll/pitch/yaw) to better fit the robot silhouette to the real arms,
using pixel darkness under the mask as the fit signal (the robot is black; a well-placed
mask should sit on low-intensity pixels).

Loss = MEAN grayscale pixel value under the projected mask (not raw sum -- sum is
minimized for free by shrinking the mask to nothing, which is what a first version of
this script did), averaged over the 30 evenly spaced frames that have a non-empty mask
(a frame with zero mask pixels just doesn't contribute -- no penalty). Search is bounded
tightly around the current config values (see BOUND_*) so it refines rather than wanders
to an unrelated pose.

    python scripts/optimize_bimanual_urdf.py --config configs/pnp_v5.yaml --episode episode_0000000

Prints the optimized params, then opens the same slider viewer as inspect_robot_mask.py
rendered with the optimized params.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib
for _backend in ("QtAgg", "Qt5Agg", "TkAgg", "GTK4Agg"):
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        continue
else:
    raise RuntimeError("no interactive matplotlib backend available -- pip install pyqt5")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "precompute"))
from common.config import load_config          # noqa: E402
from common.episodes import list_episodes, load_episode  # noqa: E402
from gam.perception.arm_render import URDFArmMask  # noqa: E402
from tune_bimanual_urdf import _write_urdf_with_separation, _rpy_to_R  # noqa: E402

# how far the search may stray from the current config values -- keep tight, this is a
# refinement, not a free re-placement
FIXED_X = 0.0          # origin x is not tuned -- pinned to the original default
BOUND_DIST = 0.10      # m, on left/right arm distance from origin
BOUND_XYZ = 0.08        # m, on global origin x/y/z
BOUND_DEG = 15.0        # deg, on global roll/pitch/yaw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--method", choices=["powell", "de"], default="powell",
                     help="powell = fast local search, can get stuck; "
                          "de = differential_evolution, global within the same bounds, slower")
    ap.add_argument("--init-dist-left", type=float, default=0.2272, help="starting left arm distance from origin (m)")
    ap.add_argument("--init-dist-right", type=float, default=0.2504, help="starting right arm distance from origin (m)")
    ap.add_argument("--color", default="255,0,0")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--fixed-params", default="", help="comma list left,right,y,z,roll,pitch,yaw (origin x "
                                                         f"is fixed at {FIXED_X}, not tuned) -- skip optimizing, "
                                                         "just render+show these")
    ap.add_argument("--init-params", default="", help="comma list left,right,y,z,roll,pitch,yaw -- "
                                                        "seed the search here instead of config defaults "
                                                        "(e.g. continue from a previous run's output)")
    args = ap.parse_args()
    color = tuple(int(c) for c in args.color.split(","))

    cfg = load_config(args.config)
    r = cfg["perception"]["robot"]
    src_urdf = r["urdf"]
    T0 = np.asarray(r["T_base_root"], np.float64)
    x0_xyz = T0[:3, 3].tolist()
    R0 = T0[:3, :3]
    yaw0 = np.degrees(np.arctan2(R0[1, 0], R0[0, 0]))

    path = next(p for p in list_episodes(cfg)
                if os.path.splitext(os.path.basename(p))[0] == args.episode)
    ep = load_episode(path, cfg)
    idxs = np.linspace(0, ep.T - 1, args.n_frames, dtype=int)
    rgbs = [ep.rgb(int(t)) for t in idxs]
    grays = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float64) for im in rgbs]
    Ks = [ep.K[t].astype(np.float64) for t in idxs]
    Es = [ep.E[t].astype(np.float64) for t in idxs]
    states = [ep.state[t] for t in idxs]
    ep.close()

    mesh_cache = {}

    def get_mask_fn(left_y, right_y):
        key = (round(left_y, 3), round(right_y, 3))
        if key not in mesh_cache:
            urdf_path = _write_urdf_with_separation(src_urdf, left_y, right_y)
            mesh_cache[key] = URDFArmMask(urdf_path, r["joint_names"], T_base_root=np.eye(4),
                                          link_filter=r.get("link_filter"), max_faces=int(r.get("max_faces", 1500)),
                                          joint_affine=r.get("joint_affine"), joint_mirror=r.get("joint_mirror"))
        return mesh_cache[key]

    eval_count = [0]

    def loss(params):
        left, right, y, z, roll, pitch, yaw = params
        mask_fn = get_mask_fn(left, right)
        T = np.eye(4)
        T[:3, :3] = _rpy_to_R(roll, pitch, yaw)
        T[:3, 3] = [FIXED_X, y, z]
        mask_fn.T_base_root = T
        per_frame = []
        for i in range(len(idxs)):
            H, W = rgbs[i].shape[:2]
            mask = mask_fn(states[i], Ks[i], Es[i], H, W).astype(bool)
            if mask.any():                      # no penalty -- frames with an empty mask
                per_frame.append(float(grays[i][mask].mean()))   # just don't contribute
        avg = float(np.mean(per_frame)) if per_frame else 255.0

        eval_count[0] += 1
        if eval_count[0] % 5 == 0:
            print(f"[optimize] eval {eval_count[0]}: loss={avg:.2f}  params={np.round(params, 4).tolist()}")
        return avg

    if args.init_params:
        x0 = [float(v) for v in args.init_params.split(",")]
        print(f"[optimize] seeding search from --init-params: {x0}")
    else:
        x0 = [args.init_dist_left, args.init_dist_right] + x0_xyz[1:] + [0.0, 0.0, yaw0]
    bounds = [
        (x0[0] - BOUND_DIST, x0[0] + BOUND_DIST),
        (x0[1] - BOUND_DIST, x0[1] + BOUND_DIST),
        (x0[2] - BOUND_XYZ, x0[2] + BOUND_XYZ),
        (x0[3] - BOUND_XYZ, x0[3] + BOUND_XYZ),
        (x0[4] - BOUND_DEG, x0[4] + BOUND_DEG),
        (x0[5] - BOUND_DEG, x0[5] + BOUND_DEG),
        (x0[6] - BOUND_DEG, x0[6] + BOUND_DEG),
    ]
    if args.fixed_params:
        left, right, y, z, roll, pitch, yaw = [float(v) for v in args.fixed_params.split(",")]
        print(f"[optimize] using fixed params (skipping search): {args.fixed_params}")
    else:
        start_loss = loss(x0)
        print(f"[optimize] starting loss (mean px, lower=darker=better): {start_loss:.2f}")
        if args.method == "de":
            res = differential_evolution(loss, bounds, x0=x0, maxiter=args.maxiter,
                                          popsize=15, tol=1e-3, seed=0, polish=True,
                                          updating="deferred", workers=1)
        else:
            res = minimize(loss, x0, method="Powell", bounds=bounds,
                            options={"maxiter": args.maxiter, "xtol": 1e-3, "ftol": 1e-2})
        print(f"[optimize] final loss: {res.fun:.2f} (started {start_loss:.2f})")
        left, right, y, z, roll, pitch, yaw = res.x
    print(f"[optimize] left_arm_dist_m={left:.4f}  right_arm_dist_m={right:.4f}")
    T = np.eye(4); T[:3, :3] = _rpy_to_R(roll, pitch, yaw); T[:3, 3] = [FIXED_X, y, z]
    print(f"[optimize] T_base_root={T.tolist()}")

    # render final frames in the same slider viewer as inspect_robot_mask.py
    mask_fn = get_mask_fn(left, right)
    mask_fn.T_base_root = T
    blended_rgb, pxcounts = [], []
    for i in range(len(idxs)):
        H, W = rgbs[i].shape[:2]
        mask = mask_fn(states[i], Ks[i], Es[i], H, W)
        bgr = cv2.cvtColor(rgbs[i], cv2.COLOR_RGB2BGR)
        overlay = bgr.copy(); overlay[mask.astype(bool)] = color
        blended = cv2.addWeighted(overlay, args.alpha, bgr, 1 - args.alpha, 0)
        blended_rgb.append(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        pxcounts.append(int(mask.sum()))
        print(f"[optimize] final frame {idxs[i]}: {pxcounts[-1]} mask px")

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.15)
    im = ax.imshow(blended_rgb[0])
    ax.set_title(f"OPTIMIZED — frame {idxs[0]}  ({pxcounts[0]} px)")
    ax.axis("off")
    ax_s = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_s, "sample", 0, len(idxs) - 1, valinit=0, valstep=1)

    def update(val):
        i = int(slider.val)
        im.set_data(blended_rgb[i])
        ax.set_title(f"OPTIMIZED — frame {idxs[i]}  ({pxcounts[i]} px)")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()
