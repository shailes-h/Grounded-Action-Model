"""Camera / point-cloud geometry shared by grounding and the observation builder.

Conventions
    K            (3,3) pinhole intrinsics of the RGB-D camera (pixels).
    E            (3,4) world->camera extrinsics [R | t] (OpenCV camera: +z forward,
                 +y down). "world" is the robot base frame the policy was trained in.
    depth        (H,W) metres, 0 = invalid.
    object       dict(p=(3,) world centre, extents=(3,) full OBB size, R=(3,3) columns =
                 OBB axes in world). Axis-aligned boxes use R = I.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

INBOX_SCALE, INBOX_PAD = 1.05, 0.01      # OBB crop tolerance for object points


def cam_to_world_to_E(T_cam2world: np.ndarray) -> np.ndarray:
    """4x4 camera->world pose -> (3,4) world->camera extrinsics."""
    T = np.asarray(T_cam2world, np.float64).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]
    return np.concatenate([R.T, (-R.T @ t)[:, None]], 1).astype(np.float32)


def project(p_world: np.ndarray, K: np.ndarray, E: np.ndarray) -> Optional[np.ndarray]:
    """world point -> (u,v) pixel; None if behind the camera."""
    E = np.asarray(E, np.float64).reshape(3, 4)
    pc = E[:, :3] @ np.asarray(p_world, np.float64) + E[:, 3]
    if pc[2] <= 1e-6:
        return None
    uvw = np.asarray(K, np.float64) @ pc
    return np.array([uvw[0] / uvw[2], uvw[1] / uvw[2]])


def backproject(u: float, v: float, z: float, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    """pixel + depth (m) -> world point."""
    K = np.asarray(K, np.float64); E = np.asarray(E, np.float64).reshape(3, 4)
    pc = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    return (E[:, :3].T @ (pc - E[:, 3])).astype(np.float32)


def depth_to_world(depth_m: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    """(H,W) depth in metres -> (N,3) world points for all valid pixels (row-major)."""
    d = np.asarray(depth_m, np.float32)
    H, W = d.shape
    us = np.tile(np.arange(W, dtype=np.float32), H)
    vs = np.repeat(np.arange(H, dtype=np.float32), W)
    z = d.reshape(-1)
    ok = z > 1e-4
    K = np.asarray(K, np.float32)
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    Pc = np.stack([x, y, z], 1)[ok]
    E = np.asarray(E, np.float32)
    return (Pc - E[:, 3]) @ E[:, :3]


def depth_to_world_grid(depth_m: np.ndarray, K: np.ndarray, E: np.ndarray):
    """(H,W) depth -> (H,W,3) world points + (H,W) validity (dense, pixel aligned)."""
    d = np.asarray(depth_m, np.float64)
    H, W = d.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    K = np.asarray(K, np.float64); E = np.asarray(E, np.float64).reshape(3, 4)
    Pc = np.stack([(uu - K[0, 2]) / K[0, 0] * d, (vv - K[1, 2]) / K[1, 1] * d, d], -1)
    Pw = (Pc - E[:, 3]) @ E[:, :3]
    return Pw, d > 1e-4


def obb_rot6d(R: np.ndarray) -> np.ndarray:
    """OBB rotation (columns = axes) -> 6-d rotation (first two columns)."""
    U = np.asarray(R, np.float32).reshape(3, 3)
    return np.concatenate([U[:, 0], U[:, 1]]).astype(np.float32)


def crop_obb(pts: np.ndarray, center, extents, R, n_pts: int, seed: int = 0):
    """Points inside an (inflated) OBB, subsampled to n_pts. Returns (pts(n,3), mask(n,))."""
    out = np.zeros((n_pts, 3), np.float32)
    msk = np.zeros((n_pts,), bool)
    if pts is None or len(pts) == 0:
        return out, msk
    U = np.asarray(R, np.float32).reshape(3, 3)
    loc = (np.asarray(pts, np.float32) - np.asarray(center, np.float32)) @ U
    inb = np.all(np.abs(loc) <= np.asarray(extents, np.float32) / 2 * INBOX_SCALE + INBOX_PAD, 1)
    v = pts[inb]
    if len(v) > n_pts:
        v = v[np.random.RandomState(int(seed)).choice(len(v), n_pts, replace=False)]
    out[:len(v)] = v
    msk[:len(v)] = True
    return out, msk


def box2d_from_obb(center, extents, R, E, K) -> np.ndarray:
    """3D OBB -> 2D pixel box [x0,y0,x1,y1] (zeros if not in front of the camera)."""
    U = np.asarray(R, np.float32).reshape(3, 3)
    sgn = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], np.float32)
    corners = np.asarray(center, np.float32) + (sgn * np.asarray(extents, np.float32) / 2) @ U.T
    E = np.asarray(E, np.float32); K = np.asarray(K, np.float32)
    pc = corners @ E[:, :3].T + E[:, 3]
    if (pc[:, 2] > 0.01).sum() < 4:
        return np.zeros(4, np.float32)
    z = np.clip(pc[:, 2], 1e-4, None)
    uv = (pc[:, :2] / z[:, None]) @ K[:2, :2].T + K[:2, 2]
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()], np.float32)


def build_slot_obs(pcd_world: np.ndarray, objects: Dict[str, dict], slot_names: List[str],
                   n_pts: int, E=None, K=None, seed: int = 0) -> Dict[str, np.ndarray]:
    """Per-slot geometry the policy consumes.

    Returns pcd_pts (S,n,3) world points inside each object's OBB, pcd_mask (S,n),
    xyz (S,3) centres, boxgeo (S,9) = [extents(3) | rot6d(6)], box2d (S,4) pixel boxes.
    A slot whose object is missing stays all-zero with mask False (= empty slot).
    """
    S = len(slot_names)
    pp = np.zeros((S, n_pts, 3), np.float32)
    pm = np.zeros((S, n_pts), bool)
    xyz = np.zeros((S, 3), np.float32)
    bg = np.zeros((S, 9), np.float32)
    b2 = np.zeros((S, 4), np.float32)
    for k, nm in enumerate(slot_names):
        o = objects.get(nm)
        if o is None:
            continue
        c = np.asarray(o["p"], np.float32)
        ext = np.asarray(o["extents"], np.float32)
        R = np.asarray(o["R"], np.float32).reshape(3, 3)
        pp[k], pm[k] = crop_obb(pcd_world, c, ext, R, n_pts, seed)
        xyz[k] = c
        bg[k, 0:3] = ext
        bg[k, 3:9] = obb_rot6d(R)
        if E is not None and K is not None:
            b2[k] = box2d_from_obb(c, ext, R, E, K)
    return dict(pcd_pts=pp, pcd_mask=pm, xyz=xyz, boxgeo=bg, box2d=b2)


def pad_slots(d: Dict[str, np.ndarray], min_slots: int) -> Dict[str, np.ndarray]:
    """Append empty slots (mask False, zeros) so every array has >= min_slots slots."""
    S = d["xyz"].shape[-2]
    if S >= min_slots:
        return d
    pad = min_slots - S
    out = dict(d)
    for k, a in d.items():
        a = np.asarray(a)
        if k == "pcd_pts":
            out[k] = np.concatenate([a, np.zeros(a.shape[:-3] + (pad,) + a.shape[-2:], a.dtype)], -3)
        elif k in ("pcd_mask", "xyz", "boxgeo", "box2d"):
            out[k] = np.concatenate([a, np.zeros(a.shape[:-2] + (pad, a.shape[-1]), a.dtype)], -2)
    return out


def object_from_box(box_xyxy, depth_m, K, E, extents_min=0.03, extents_max=0.40):
    """2D pixel box + depth -> axis-aligned object (p, extents, R=I) from the box interior
    points (median-z window rejects box corners that see the background)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy]
    H, W = depth_m.shape
    x0, x1 = max(x0, 0), min(x1, W - 1); y0, y1 = max(y0, 0), min(y1, H - 1)
    if x1 <= x0 or y1 <= y0:
        return None
    Pw, ok = depth_to_world_grid(depth_m, K, E)
    sub = ok[y0:y1 + 1, x0:x1 + 1]
    if sub.sum() < 10:
        return None
    pts = Pw[y0:y1 + 1, x0:x1 + 1][sub]
    zc = np.asarray(depth_m, np.float64)[y0:y1 + 1, x0:x1 + 1][sub]
    zmed = np.median(zc)
    keep = np.abs(zc - zmed) < 0.15                     # 15 cm band around the object depth
    if keep.sum() >= 10:
        pts = pts[keep]
    lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
    ext = np.clip(hi - lo, extents_min, extents_max).astype(np.float32)
    return {"p": ((lo + hi) / 2).astype(np.float32), "extents": ext, "R": np.eye(3, dtype=np.float32)}
