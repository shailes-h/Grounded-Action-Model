"""Window dataset over the precomputed cache (one .npz per episode)."""
from __future__ import annotations

import glob
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class CacheDataset(Dataset):
    """Each sample: observation at frame t (state history included) + the next `horizon`
    actions (clamped at the episode end, repeating the last action)."""

    def __init__(self, cache_dir: str, episodes: List[str] | None = None, horizon: int = 16,
                 state_hist: int = 2):
        paths = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
        if episodes is not None:
            paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in episodes]
        assert paths, f"no cache episodes in {cache_dir}"
        self.eps = [dict(np.load(p, allow_pickle=True)) for p in paths]
        self.names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        self.horizon = horizon
        self.state_hist = state_hist
        self.index = [(e, t) for e, ep in enumerate(self.eps) for t in range(len(ep["state"]))]
        self.action_dim = self.eps[0]["action"].shape[-1]
        self.image_wh = tuple(float(v) for v in self.eps[0]["image_wh"])

    def __len__(self):
        return len(self.index)

    def all_actions(self) -> np.ndarray:
        return np.concatenate([ep["action"] for ep in self.eps], 0)

    def obs_at(self, e: int, t: int) -> dict:
        ep = self.eps[e]
        hist = [ep["state"][max(t - k, 0)] for k in range(self.state_hist - 1, -1, -1)]
        return {
            "patch_tokens": torch.from_numpy(ep["patches"][t].astype(np.float32)),
            "box2d": torch.from_numpy(ep["box2d"][t]),
            "xyz": torch.from_numpy(ep["xyz"][t]),
            "boxgeo": torch.from_numpy(ep["boxgeo"][t]),
            "pcd_pts": torch.from_numpy(ep["pcd_pts"][t].astype(np.float32)),
            "pcd_mask": torch.from_numpy(ep["pcd_mask"][t]),
            "state": torch.from_numpy(np.concatenate(hist).astype(np.float32)),
            "lang": torch.from_numpy(np.asarray(ep["lang"], np.float32)) if "lang" in ep
                    else torch.zeros(768),
        }

    def __getitem__(self, i: int):
        e, t = self.index[i]
        ep = self.eps[e]
        T = len(ep["state"])
        idx = np.clip(np.arange(t, t + self.horizon), 0, T - 1)
        return {"obs": self.obs_at(e, t),
                "action": torch.from_numpy(ep["action"][idx].astype(np.float32))}


def fit_limits_normalizer(actions: np.ndarray, out_max=1.0, out_min=-1.0, eps=1e-4):
    """x_norm = x * scale + offset mapping per-dim [min,max] -> [-1,1] (constant dims -> mid)."""
    a = torch.from_numpy(np.asarray(actions, np.float32))
    lo, hi = a.min(0).values, a.max(0).values
    rng = hi - lo
    ignore = rng < eps
    rng = torch.where(ignore, torch.full_like(rng, out_max - out_min), rng)
    scale = (out_max - out_min) / rng
    offset = out_min - scale * lo
    offset[ignore] = (out_max + out_min) / 2 - lo[ignore]
    return scale, offset
