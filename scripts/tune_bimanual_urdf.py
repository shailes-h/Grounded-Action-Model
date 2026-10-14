#!/usr/bin/env python3
"""Interactively tune the bimanual YAM URDF's arm separation and global (base_link ->
policy base frame) pose against one episode frame, before committing values to the config.

    python scripts/tune_bimanual_urdf.py --config configs/pnp_v5.yaml --episode episode_0000000

Sliders: arm separation (m), global origin x/y/z (m), roll/pitch/yaw (deg).
Move sliders, hit "apply" -> re-renders the mask on the currently selected frame.
The frame slider re-renders with the last-applied params (no URDF reload needed).

Prints the final T_base_root matrix + arm_separation_m on exit -- copy those into
configs/*.yaml's perception.robot.T_base_root / the urdf regeneration call.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import cv2
import numpy as np
import matplotlib
for _backend in ("TkAgg", "QtAgg", "Qt5Agg", "GTK4Agg"):
    try:
        matplotlib.use(_backend, force=True)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.close("all")
        break
    except Exception:
        continue
else:
    raise RuntimeError(
        "no interactive matplotlib backend available -- "
        "install python3-tk (TkAgg) or pyqt5 (QtAgg)"
    )
from matplotlib.widgets import Slider, Button

FIXED_X = 0.0  # origin x is not tuned -- pinned to the original default

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "precompute"))
from common.config import load_config          # noqa: E402
from common.episodes import list_episodes, load_episode  # noqa: E402
from gam.perception.arm_render import URDFArmMask  # noqa: E402


def _rpy_to_R(roll, pitch, yaw):
    r, p, y = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _write_urdf_with_separation(src_urdf: str, left_y: float, right_y: float) -> str:
    """Rewrite left_mount/right_mount joint origins to their own distance-from-origin
    along Y (independent, not symmetric); write next to the source file so relative
    mesh paths keep resolving."""
    xml = open(src_urdf).read()
    for side, y in (("left_mount", left_y), ("right_mount", -right_y)):
        xml = re.sub(
            rf'(<joint name="{side}"[^>]*>\s*<origin xyz=")[^"]*(")',
            rf'\g<1>0 {y} 0\g<2>', xml)
    out_path = os.path.join(os.path.dirname(src_urdf), "_tune_tmp.urdf")
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--color", default="255,0,0")
    ap.add_argument("--init-pose", default="", help="comma list left,right,y,z,roll,pitch,yaw -- "
                                                      "seed the sliders here instead of config defaults")
    args = ap.parse_args()
    color = tuple(int(c) for c in args.color.split(","))

    cfg = load_config(args.config)
    r = cfg["perception"]["robot"]
    src_urdf = r["urdf"]
    T0 = np.asarray(r["T_base_root"], np.float64)
    init_xyz = T0[:3, 3].tolist()
    init_R = T0[:3, :3]
    init_yaw = np.degrees(np.arctan2(init_R[1, 0], init_R[0, 0]))
    init_left, init_right, init_roll, init_pitch = 0.2272, 0.2504, 0.2492, 0.4002

    if args.init_pose:
        vals = [float(v) for v in args.init_pose.replace(" ", "").split(",")]
        init_left, init_right, init_xyz[1], init_xyz[2], init_roll, init_pitch, init_yaw = vals
        print(f"[tune] seeding sliders from --init-pose: {vals}")

    path = next(p for p in list_episodes(cfg)
                if os.path.splitext(os.path.basename(p))[0] == args.episode)
    ep = load_episode(path, cfg)
    idxs = np.linspace(0, ep.T - 1, args.n_frames, dtype=int)
    rgbs = [ep.rgb(int(t)) for t in idxs]
    Ks = [ep.K[t].astype(np.float64) for t in idxs]
    Es = [ep.E[t].astype(np.float64) for t in idxs]
    states = [ep.state[t] for t in idxs]

    state = {"mask_fn": None, "T_base_root": T0.copy(), "frame_i": 0}

    def rebuild_mask_fn(left_y, right_y):
        urdf_path = _write_urdf_with_separation(src_urdf, left_y, right_y)
        state["mask_fn"] = URDFArmMask(urdf_path, r["joint_names"], T_base_root=np.eye(4),
                                        link_filter=r.get("link_filter"), max_faces=int(r.get("max_faces", 1500)),
                                        joint_affine=r.get("joint_affine"), joint_mirror=r.get("joint_mirror"))

    def render(i):
        rgb = rgbs[i]; H, W = rgb.shape[:2]
        mask_fn = state["mask_fn"]
        mask_fn.T_base_root = state["T_base_root"]
        mask = mask_fn(states[i], Ks[i], Es[i], H, W)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = bgr.copy(); overlay[mask.astype(bool)] = color
        blended = cv2.addWeighted(overlay, args.alpha, bgr, 1 - args.alpha, 0)
        return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), int(mask.sum())

    fig, ax = plt.subplots(figsize=(9, 8.5))
    plt.subplots_adjust(bottom=0.55)
    rebuild_mask_fn(init_left, init_right)
    img, npx = render(0)
    im = ax.imshow(img)
    ax.set_title(f"frame {idxs[0]}  ({npx} px)")
    ax.axis("off")

    def S(y, label, vmin, vmax, vinit):
        return Slider(plt.axes([0.25, y, 0.55, 0.025]), label, vmin, vmax, valinit=vinit)

    s_frame = S(0.47, "frame idx", 0, len(idxs) - 1, 0)
    s_sep_l = S(0.41, "left arm dist from origin (m)", 0.0, 1.0, init_left)
    s_sep_r = S(0.35, "right arm dist from origin (m)", 0.0, 1.0, init_right)
    s_y = S(0.29, "origin y (m)", -1.0, 1.0, init_xyz[1])
    s_z = S(0.23, "origin z (m)", -1.0, 1.0, init_xyz[2])
    s_roll = S(0.17, "roll (deg)", -180, 180, init_roll)
    s_pitch = S(0.11, "pitch (deg)", -180, 180, init_pitch)
    s_yaw = S(0.06, "yaw (deg)", -180, 180, init_yaw)
    b_apply = Button(plt.axes([0.83, 0.20, 0.12, 0.05]), "apply")

    def on_frame(val):
        i = int(s_frame.val); state["frame_i"] = i
        img, npx = render(i)
        im.set_data(img)
        ax.set_title(f"frame {idxs[i]}  ({npx} px)")
        fig.canvas.draw_idle()

    def on_apply(event):
        rebuild_mask_fn(s_sep_l.val, s_sep_r.val)
        R = _rpy_to_R(s_roll.val, s_pitch.val, s_yaw.val)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [FIXED_X, s_y.val, s_z.val]
        state["T_base_root"] = T
        on_frame(None)
        print(f"[tune] left_arm_dist_m={s_sep_l.val:.4f}  right_arm_dist_m={s_sep_r.val:.4f}")
        print(f"[tune] T_base_root={T.tolist()}")

    s_frame.on_changed(on_frame)
    b_apply.on_clicked(on_apply)
    plt.show()
    ep.close()


if __name__ == "__main__":
    main()
