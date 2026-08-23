"""Precompute the training cache: for every episode and frame, run the perception stack
(WildDet3D patch tokens + boxes, URDF robot silhouette, depth geometry) and store
everything the policy trains on. Output: one <episode>.npz per episode in train.cache_dir.

    python precompute/precompute.py --config configs/my_task.yaml [--ep-start 0 --ep-end 999]

Sharding: run several jobs with disjoint --ep-start/--ep-end; finished episodes are skipped.
Cost: one WildDet3D forward per frame per slot (~0.5 s on a modern GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.config import load_config  # noqa: E402
from common.episodes import list_episodes, load_episode  # noqa: E402


def build_arm_mask_fn(cfg):
    r = cfg["perception"].get("robot") or None
    if not r:
        print("[precompute] WARNING: no robot section — patch masks will not contain the robot "
              "(the policy learns without it; keep eval identical)")
        return None
    from gam.perception.arm_render import URDFArmMask
    m = URDFArmMask(r["urdf"], r["joint_names"], T_base_root=np.asarray(r["T_base_root"], np.float64),
                    link_filter=r.get("link_filter"), max_faces=int(r.get("max_faces", 1500)),
                    joint_affine=r.get("joint_affine"), joint_mirror=r.get("joint_mirror"))
    print(f"[precompute] URDF silhouette: {len(m._nodes)} geometries, {m.n_faces} faces")
    return m


def geometry_from_detection(det, depth_m, K, E, source: str):
    """WildDet3D detection -> object dict, by config `perception.geometry_source`:
    box2d_depth = depth backprojection inside the predicted 2D box (axis-aligned);
    box3d       = the predicted oriented 3D box. Falls back to the other source."""
    from gam.perception.geometry import object_from_box
    obj = None
    if source == "box3d" and det.center is not None:
        obj = {"p": det.center, "extents": det.extents, "R": det.R}
    if obj is None and det.box2d[2] > det.box2d[0]:
        obj = object_from_box(det.box2d, depth_m, K, E)
    if obj is None and det.center is not None:
        obj = {"p": det.center, "extents": det.extents, "R": det.R}
    return obj


def process_episode(ep, ext, arm_fn, cfg, out_path):
    from gam.perception.geometry import build_slot_obs, depth_to_world
    S = len(cfg["data"]["slots"])
    n_pts = int(cfg["perception"].get("n_points", 512))
    geom_src = cfg["perception"].get("geometry_source", "box2d_depth")
    G2 = int(cfg["perception"].get("patch_grid", 24)) ** 2
    T = ep.T
    patches = np.zeros((T, G2, 1024), np.float16)
    box2d = np.zeros((T, S, 4), np.float32)
    xyz = np.zeros((T, S, 3), np.float32)
    boxgeo = np.zeros((T, S, 9), np.float32)
    pcd_pts = np.zeros((T, S, n_pts, 3), np.float16)
    pcd_mask = np.zeros((T, S, n_pts), bool)
    t0 = time.time()
    for t in range(T):
        rgb = ep.rgb(t)
        dep = ep.depth_m(t)
        K, E = ep.K[t].astype(np.float64), ep.E[t].astype(np.float64)
        H, W = rgb.shape[:2]
        arm = arm_fn(ep.state[t], K, E, H, W) if arm_fn is not None else None
        cl = ep.clicks[t]
        valid = [i for i in range(S) if np.isfinite(cl[i]).all()
                 and 0 <= cl[i][0] < W and 0 <= cl[i][1] < H]
        pr = ext(rgb, dep, K, E, [tuple(cl[i]) for i in valid], arm_mask=arm)
        objs = {}
        for j, i in enumerate(valid):
            obj = geometry_from_detection(pr.detections[j], dep, K, E, geom_src)
            if obj is not None:
                objs[cfg["data"]["slots"][i]] = obj
        pcd = depth_to_world(dep, K, E)
        g = build_slot_obs(pcd, objs, cfg["data"]["slots"], n_pts, E=E, K=K, seed=t)
        patches[t] = pr.patches.numpy().astype(np.float16)
        box2d[t], xyz[t], boxgeo[t] = g["box2d"], g["xyz"], g["boxgeo"]
        pcd_pts[t] = g["pcd_pts"].astype(np.float16)
        pcd_mask[t] = g["pcd_mask"]
        if t % 25 == 0 or t == T - 1:
            print(f"  frame {t + 1}/{T}  slots-found={len(objs)}/{S}  "
                  f"({(time.time() - t0) / (t + 1):.2f} s/frame)", flush=True)
    H0, W0 = ep.rgb(0).shape[:2]
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, state=ep.state, action=ep.action, patches=patches, box2d=box2d,
                        xyz=xyz, boxgeo=boxgeo, pcd_pts=pcd_pts, pcd_mask=pcd_mask,
                        clicks=ep.clicks, image_wh=np.array([W0, H0], np.float32),
                        instruction=np.str_(ep.instruction),
                        **({"lang": ep.lang_emb} if getattr(ep, "lang_emb", None) is not None else {}))
    os.replace(tmp, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ep-start", type=int, default=0)
    ap.add_argument("--ep-end", type=int, default=10**9)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = cfg["train"]["cache_dir"]
    os.makedirs(out_dir, exist_ok=True)
    eps = list_episodes(cfg)[args.ep_start:args.ep_end]
    from gam.perception.patch_tokens import PatchTokenExtractor
    ext = PatchTokenExtractor(cfg["perception"]["wilddet_ckpt"],
                              channel_swap=bool(cfg["perception"].get("image_channel_swap", False)),
                              grid=int(cfg["perception"].get("patch_grid", 24)),
                              bf16=bool(cfg["perception"].get("bf16", True)),
                              cell_min_overlap=float(cfg["perception"].get("cell_min_overlap", 0.30)))
    arm_fn = build_arm_mask_fn(cfg)
    for i, path in enumerate(eps):
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(out_dir, name + ".npz")
        if os.path.exists(out):
            print(f"[{i + 1}/{len(eps)}] {name}: exists, skip")
            continue
        print(f"[{i + 1}/{len(eps)}] {name}")
        ep = load_episode(path, cfg)
        lang_emb = None
        if cfg.get("text"):                        # whole-sentence T5 token (train.use_language)
            from text.language_clicks import sentence_embedding
            lang_emb = sentence_embedding(cfg, ep.instruction).astype(np.float32)
        if cfg["data"]["clicks"]["source"] == "language":
            from text.language_clicks import clicks_from_language
            px, queries = clicks_from_language(cfg, ep.rgb(0), ep.instruction)
            ep.clicks[:] = px[None]
            print(f"    language clicks: {dict(zip(cfg['data']['slots'], queries))}")
        ep.lang_emb = lang_emb
        process_episode(ep, ext, arm_fn, cfg, out)
        ep.close()
    json.dump({"slots": cfg["data"]["slots"], "n_points": cfg["perception"].get("n_points", 512),
               "patch_grid": int(cfg["perception"].get("patch_grid", 24)),
               "geometry_source": cfg["perception"].get("geometry_source", "box2d_depth")},
              open(os.path.join(out_dir, "meta.json"), "w"))
    print("PRECOMPUTE_DONE")


if __name__ == "__main__":
    main()
