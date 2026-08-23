"""Episode reader driven by the config's key map — adapt the YAML, not the code.

An episode is one HDF5 file with, per frame, the head camera RGB-D + calibration, the
robot state and the action, plus a pixel click per object slot (either recorded per
frame, or projected from known 3D object centres). See configs/example.yaml.
"""
from __future__ import annotations

import glob
import os
import json
from dataclasses import dataclass
from typing import List, Optional

import cv2
import h5py
import numpy as np

from gam.perception.geometry import project


@dataclass
class Episode:
    path: str
    T: int
    state: np.ndarray            # (T, state_dim)
    action: np.ndarray           # (T, action_dim)
    K: np.ndarray                # (T, 3, 3)
    E: np.ndarray                # (T, 3, 4) world(base)->camera, OpenCV
    clicks: np.ndarray           # (T, S, 2) pixel click per slot (NaN = missing that frame)
    instruction: str = ""

    _f: Optional[h5py.File] = None
    _rgb_key: str = ""
    _depth_key: str = ""
    _depth_scale: float = 1.0

    def rgb(self, t: int) -> np.ndarray:
        """(H,W,3) uint8 RGB."""
        v = self._f[self._rgb_key][t]
        if isinstance(v, np.ndarray) and v.ndim == 3:
            return v
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(np.asarray(v).tobytes()
                                                       if not isinstance(v, (bytes, np.bytes_)) else v,
                                                       np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    def depth_m(self, t: int) -> np.ndarray:
        """(H,W) float32 metres."""
        return np.asarray(self._f[self._depth_key][t], np.float32) * self._depth_scale

    def close(self):
        if self._f is not None:
            self._f.close()


def _per_frame(a, T):
    a = np.asarray(a, np.float64)
    return a if a.ndim >= 3 or (a.ndim == 2 and a.shape[0] == T and a.shape != (3, 3)) \
        else np.repeat(a[None], T, 0)


def list_episodes(cfg: dict) -> List[str]:
    eps = sorted(glob.glob(cfg["data"]["episodes_glob"]))
    assert eps, f"no episodes match {cfg['data']['episodes_glob']}"
    return eps


def load_episode(path: str, cfg: dict) -> Episode:
    d = cfg["data"]
    f = h5py.File(path, "r")
    state = np.asarray(f[d["state_key"]], np.float32)
    action = np.asarray(f[d["action_key"]], np.float32)
    T = len(state)
    K = _per_frame(f[d["intrinsic_key"]], T).reshape(T, 3, 3)
    E_raw = _per_frame(f[d["extrinsic_key"]], T)
    if d.get("extrinsic_is_cam2base", False):                    # 4x4 camera pose -> world->cam
        E = np.zeros((T, 3, 4))
        for t in range(T):
            M = E_raw[t].reshape(4, 4)
            E[t, :, :3] = M[:3, :3].T
            E[t, :, 3] = -M[:3, :3].T @ M[:3, 3]
    else:
        E = E_raw.reshape(T, -1)[:, :12].reshape(T, 3, 4)
    slots = d["slots"]
    S = len(slots)
    clicks = np.full((T, S, 2), np.nan, np.float32)
    csrc = d["clicks"]["source"]
    if csrc == "dataset":
        c = np.asarray(f[d["clicks"]["dataset_key"]], np.float32)
        if c.ndim == 2:                                          # (S,2) frame-0 -> held constant
            clicks[:] = c[None]
        else:                                                    # (T,S,2) per-frame
            clicks[:] = c
    elif csrc == "project_centers":                              # 3D centres -> per-frame pixels
        for s_i, nm in enumerate(slots):
            key = d["clicks"]["centers_key_fmt"].format(slot=nm)
            if key not in f:
                continue
            P = _per_frame(f[key], T).reshape(T, -1)[:, :3]
            for t in range(T):
                uv = project(P[t], K[t], E[t])
                if uv is not None:
                    clicks[t, s_i] = uv
    elif csrc == "annotations":                                  # labeling/annotator.py output
        ap = os.path.join(d["clicks"]["annotations_dir"],
                          os.path.splitext(os.path.basename(path))[0] + ".json")
        if os.path.exists(ap):
            pts = json.load(open(ap)).get("points", [])
            ordered = ([p_["point"] for p_ in pts if p_["role"] == "pick"]
                       + [p_["point"] for p_ in pts if p_["role"] == "place"])
            if len(ordered) != S:
                print(f"[episodes] WARNING {os.path.basename(path)}: {len(ordered)} marks for "
                      f"{S} slots (order = picks then places, must match data.slots)")
            for s_i in range(min(S, len(ordered))):
                clicks[:, s_i] = np.asarray(ordered[s_i], np.float32)
        else:
            print(f"[episodes] WARNING no annotation for {os.path.basename(path)}")
    elif csrc == "language":
        pass          # filled by text/language_clicks.py from the instruction (frame-0, held constant)
    else:
        raise ValueError(f"unknown clicks.source {csrc!r}")
    instr = ""
    ik = d.get("instruction_key")
    if ik and ik in f:
        v = f[ik][()]
        v = v[0] if getattr(v, "shape", None) else v
        instr = v.decode() if isinstance(v, bytes) else str(v)
    ep = Episode(path=path, T=T, state=state, action=action, K=K.astype(np.float32),
                 E=E.astype(np.float32), clicks=clicks, instruction=instr)
    ep._f = f
    ep._rgb_key = d["rgb_key"]
    ep._depth_key = d["depth_key"]
    ep._depth_scale = 0.001 if d.get("depth_unit", "m") == "mm" else 1.0
    return ep
