"""Visualise PCD alignment: backproject depth → world → reproject to image.

Shows:
  - Full depth backprojection reprojected onto RGB (green dots) — alignment check
  - Per-slot point clouds from the cache projected onto RGB (coloured per slot)
  - Click positions as cross markers

Usage:
    python scripts/inspect_pcd_alignment.py --config configs/pnp_v5.yaml --episode episode_0000000 [--frame 0] [--out pcd.png]
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.config import load_config
from common.episodes import list_episodes, load_episode


def project_points(pts_world, K, E):
    """(N,3) world points → (N,2) pixel coords, (N,) validity mask."""
    E = np.asarray(E, np.float64).reshape(3, 4)
    K = np.asarray(K, np.float64)
    Pc = pts_world @ E[:, :3].T + E[:, 3]
    z = Pc[:, 2]
    ok = z > 0.01
    uv = np.zeros((len(pts_world), 2))
    uv[ok, 0] = K[0, 0] * Pc[ok, 0] / z[ok] + K[0, 2]
    uv[ok, 1] = K[1, 1] * Pc[ok, 1] / z[ok] + K[1, 2]
    return uv, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Load raw episode
    src = [p for p in list_episodes(cfg) if args.episode in p]
    assert src, f"episode {args.episode} not found"
    ep = load_episode(src[0], cfg)
    t = args.frame

    # Load cache
    cache_path = os.path.join(cfg["train"]["cache_dir"], args.episode + ".npz")
    z = np.load(cache_path)

    rgb = ep.rgb(t)
    dep = ep.depth_m(t)
    K = ep.K[t].astype(np.float64)
    E = ep.E[t].astype(np.float64)
    H, W = rgb.shape[:2]

    slot_names = cfg["data"]["slots"]
    slot_colors = [
        (0, 255, 0),    # pick_1: green
        (0, 165, 255),  # place_1: orange
        (255, 0, 0),    # pick_2: blue
        (0, 0, 255),    # place_2: red
    ]

    # === Panel 1: Full depth backprojection reprojected ===
    from gam.perception.geometry import depth_to_world
    pcd_full = depth_to_world(dep, K, E)
    uv_full, ok_full = project_points(pcd_full, K, E)

    vis1 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    uv_ok = uv_full[ok_full].astype(int)
    in_bounds = (uv_ok[:, 0] >= 0) & (uv_ok[:, 0] < W) & (uv_ok[:, 1] >= 0) & (uv_ok[:, 1] < H)
    for u, v in uv_ok[in_bounds][::3]:  # subsample for clarity
        cv2.circle(vis1, (u, v), 1, (0, 200, 0), -1)
    cv2.putText(vis1, "Depth->World->Reproject (green=aligned)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # === Panel 2: Per-slot PCD from cache reprojected ===
    vis2 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for s in range(z["pcd_pts"].shape[1]):
        pts = z["pcd_pts"][t, s].astype(np.float64)   # (512, 3) world frame
        mask = z["pcd_mask"][t, s]                      # (512,) bool
        if not mask.any():
            cv2.putText(vis2, f"{slot_names[s]}: NO PCD", (10, 60 + s * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, slot_colors[s], 2)
            continue
        pts_valid = pts[mask]
        uv, ok = project_points(pts_valid, K, E)
        uv_ok = uv[ok].astype(int)
        in_b = (uv_ok[:, 0] >= 0) & (uv_ok[:, 0] < W) & (uv_ok[:, 1] >= 0) & (uv_ok[:, 1] < H)
        for u, v in uv_ok[in_b]:
            cv2.circle(vis2, (u, v), 2, slot_colors[s], -1)
        # Draw click
        cl = z["clicks"][t, s]
        if np.isfinite(cl).all():
            cv2.drawMarker(vis2, (int(cl[0]), int(cl[1])), slot_colors[s],
                           cv2.MARKER_CROSS, 25, 2)
        # Draw box2d
        b = z["box2d"][t, s]
        if b[2] > b[0]:
            cv2.rectangle(vis2, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          slot_colors[s], 2)
        # Label
        n_pts = int(mask.sum())
        cv2.putText(vis2, f"{slot_names[s]}: {n_pts} pts", (10, 60 + s * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, slot_colors[s], 2)

    cv2.putText(vis2, "Cached slot PCD reprojected + clicks + box2d", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # === Panel 3: Depth image with slot xyz projected ===
    dep_vis = (np.clip(dep / 2.0, 0, 1) * 255).astype(np.uint8)
    vis3 = cv2.applyColorMap(dep_vis, cv2.COLORMAP_TURBO)
    for s in range(len(slot_names)):
        xyz = z["xyz"][t, s]
        if np.abs(xyz).sum() > 0:
            uv, ok = project_points(xyz.reshape(1, 3), K, E)
            if ok[0]:
                u, v = int(uv[0, 0]), int(uv[0, 1])
                cv2.drawMarker(vis3, (u, v), slot_colors[s], cv2.MARKER_STAR, 20, 3)
                cv2.putText(vis3, f"{slot_names[s]} z={xyz[2]:.3f}", (u + 10, v),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, slot_colors[s], 2)
    cv2.putText(vis3, "Depth + xyz centres projected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Stack and save
    out_path = args.out or os.path.join(cfg["train"]["cache_dir"],
                                        f"{args.episode}_pcd_frame{t}.png")
    combined = np.vstack([vis1, vis2, vis3])
    cv2.imwrite(out_path, combined)
    print(f"Wrote {out_path} ({combined.shape[1]}x{combined.shape[0]})")
    print(f"\nCalibration used:")
    print(f"  K = {K.tolist()}")
    print(f"  E = {E.tolist()}")
    print(f"  Depth range: [{dep[dep > 0].min():.4f}, {dep[dep > 0].max():.4f}] m")
    print(f"  Full PCD: {len(pcd_full)} points")

    ep.close()


if __name__ == "__main__":
    main()
