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

import cv2
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


def _depth_K(K, rgb_shape, depth_shape):
    """RGB 内参 -> 深度分辨率内参 (深度常为 RGB 的 1/2 长宽; 像素中心约定).
    返回 (KD, dscale). 分辨率相同则原样返回."""
    if tuple(depth_shape[:2]) == tuple(rgb_shape[:2]):
        return K, 1.0
    sy = depth_shape[0] / float(rgb_shape[0]); sx = depth_shape[1] / float(rgb_shape[1])
    KD = K.copy()
    KD[0, 0] *= sx; KD[1, 1] *= sy
    KD[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    KD[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return KD, sx


def _obfb_kwargs(z_band):
    """老版本 gam_release 的 object_from_box 没有 z_band 参数 -> 自动降级 (兼容垫片)."""
    import inspect
    from gam.perception.geometry import object_from_box
    return {"z_band": z_band} if "z_band" in inspect.signature(object_from_box).parameters else {}


def geometry_from_detection(det, depth_m, K, E, source: str, z_band: float = 0.15):
    """WildDet3D detection -> object dict, by config `perception.geometry_source`:
    box2d_depth = depth backprojection inside the predicted 2D box (axis-aligned);
    box3d       = the predicted oriented 3D box. Falls back to the other source."""
    from gam.perception.geometry import object_from_box
    obj = None
    if source == "box3d" and det.center is not None:
        obj = {"p": det.center, "extents": det.extents, "R": det.R}
    if obj is None and det.box2d[2] > det.box2d[0]:
        obj = object_from_box(det.box2d, depth_m, K, E, **_obfb_kwargs(z_band))
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
    # --- 传感器深度口径 (perception.sensor_depth: true, 0829 定稿) ---------------------
    # 1. 检测器永远喂零深度 (纯单目, 与 wilddet 训练时的推理口径一致); 真实深度只参与点云几何.
    # 2. 几何深度先清洗: 抠掉机器人剪影像素 + 桌面及以下 (真实深度锐利, 手指与被夹物体同深度、
    #    桌面/背后容器都会被 object_from_box 的中值带圈进 OBB -> 框和点云膨胀).
    # 3. cache 的 box2d 直接存检测框 (紧); OBB 只用于 xyz/boxgeo/pcd.
    # 4. 深度分辨率可低于 RGB (常见 1/2): 几何用按实际形状缩放的内参 KD, 框坐标乘 dscale.
    sensor_depth = bool(cfg["perception"].get("sensor_depth", False))
    z_band = float(cfg["perception"].get("z_band", 0.15))
    table_z = None
    if sensor_depth:
        from gam.perception.geometry import depth_to_world_grid
        d0 = ep.depth_m(0)
        K0 = ep.K[0].astype(np.float64)
        KD0, ds0 = _depth_K(K0, ep.rgb(0).shape, d0.shape)
        P0, ok0 = depth_to_world_grid(d0, KD0, ep.E[0].astype(np.float64))
        hz, ez = np.histogram(P0[ok0][:, 2], bins=200, range=(-0.3, 1.0))
        table_z = float(ez[hz.argmax()] + (ez[1] - ez[0]) / 2)
        print(f"  sensor-depth mode: depth {d0.shape} scale {ds0:.3f}, table z={table_z:.4f} m", flush=True)
    for t in range(T):
        rgb = ep.rgb(t)
        dep = ep.depth_m(t)
        K, E = ep.K[t].astype(np.float64), ep.E[t].astype(np.float64)
        H, W = rgb.shape[:2]
        arm = arm_fn(ep.state[t], K, E, H, W) if arm_fn is not None else None
        cl = ep.clicks[t]
        valid = [i for i in range(S) if np.isfinite(cl[i]).all()
                 and 0 <= cl[i][0] < W and 0 <= cl[i][1] < H]
        if sensor_depth:
            KD, dscale = _depth_K(K, rgb.shape, dep.shape)
            dep_geo = dep.copy()
            if arm is not None:
                am_s = cv2.resize(arm.astype(np.uint8), (dep.shape[1], dep.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
                dep_geo[am_s] = 0.0
            from gam.perception.geometry import depth_to_world_grid
            Pg, okg = depth_to_world_grid(dep_geo, KD, E)
            dep_geo[okg & (Pg[..., 2] < table_z + 0.008)] = 0.0
            det_depth = np.zeros((H, W), np.float32)          # 检测器: 零深度
        else:
            KD, dscale, dep_geo, det_depth = K, 1.0, dep, dep
        pr = ext(rgb, det_depth, K, E, [tuple(cl[i]) for i in valid], arm_mask=arm)
        objs = {}
        det_boxes = {}
        for j, i in enumerate(valid):
            det = pr.detections[j]
            if sensor_depth and det.box2d[2] > det.box2d[0]:
                import dataclasses as _dc
                det = _dc.replace(det, box2d=det.box2d * dscale)
            obj = geometry_from_detection(det, dep_geo, KD, E, geom_src, z_band=z_band)
            if obj is not None:
                objs[cfg["data"]["slots"][i]] = obj
                det_boxes[i] = pr.detections[j].box2d.astype(np.float32)
        pcd = depth_to_world(dep_geo, KD, E)
        g = build_slot_obs(pcd, objs, cfg["data"]["slots"], n_pts, E=E, K=K, seed=t)
        patches[t] = pr.patches.numpy().astype(np.float16)
        box2d[t], xyz[t], boxgeo[t] = g["box2d"], g["xyz"], g["boxgeo"]
        if sensor_depth:
            for i, b in det_boxes.items():                    # 2D 框 = 检测框 (紧), 非 OBB 反投影
                box2d[t, i] = b
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
