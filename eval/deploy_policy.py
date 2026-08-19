"""Closed-loop evaluation adapter with the common get_model / encode_obs / eval /
reset_model contract used by dual-arm manipulation benchmarks.

Your harness supplies, per step, an observation dict:
    obs["observation"]["head_camera"]{rgb (H,W,3 RGB uint8), depth (H,W, mm or m),
                                      intrinsic_cv (3,3), extrinsic_cv (3,4) base->cam}
    obs["joint_action"]["vector"]     current joint state (matches the checkpoint layout)
    obs["object_gt"]                  {name: {"p": (3,) world centre, ...}} — clicks come from
                                      projecting these centres (slot order = config data.slots)
and calls:
    model = get_model({"ckpt_path": ..., "config_path": ..., "task_name": ...})
    eval(TASK_ENV, model, observation)     # predicts 16, executes 15 with seam smoothing
    reset_model(model)                     # between episodes

The perception path (WildDet3D boxes -> geometry, URDF robot silhouette, patch masks) is
byte-identical to precompute/precompute.py — training and evaluation share the code.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.config import load_config  # noqa: E402
from precompute.precompute import build_arm_mask_fn, geometry_from_detection  # noqa: E402
from gam.model.policy import GAMPolicy  # noqa: E402
from gam.perception.geometry import build_slot_obs, depth_to_world, project  # noqa: E402
from gam.perception.patch_tokens import PatchTokenExtractor  # noqa: E402


class YamModel:
    def __init__(self, ckpt, config_path):
        self.cfg = load_config(config_path)
        self.slots = self.cfg["data"]["slots"]
        self.n_pts = int(self.cfg["perception"].get("n_points", 512))
        self.policy = GAMPolicy.from_checkpoint(ckpt)
        self.geom_src = self.cfg["perception"].get("geometry_source", "box2d_depth")
        self.ext = PatchTokenExtractor(self.cfg["perception"]["wilddet_ckpt"],
                                       channel_swap=self.policy.image_channel_swap,
                                       grid=int(round(self.policy.encoder.n_patch ** 0.5)))
        self.arm_fn = build_arm_mask_fn(self.cfg)
        self.state_hist = int(self.policy.meta.get("state_hist", 2))
        self.gen = torch.Generator(device="cuda").manual_seed(0)
        self.reset()
        print(f"[yam] ckpt={os.path.basename(ckpt)} slots={self.slots} meta={self.policy.meta}",
              flush=True)

    def reset(self):
        self.hist = []
        self.t = 0
        self._seam = None

    def push_state(self, q):
        q = np.asarray(q, np.float32).reshape(-1)
        self.hist = (self.hist + [q])[-self.state_hist:]

    @torch.no_grad()
    def act(self, obs):
        hc = obs["observation"]["head_camera"]
        rgb = np.asarray(hc["rgb"])
        dep = np.asarray(hc["depth"], np.float32)
        if dep[dep > 0].size and np.median(dep[dep > 0]) > 50:      # mm heuristic
            dep = dep / 1000.0
        K = np.asarray(hc["intrinsic_cv"], np.float64)
        E = np.asarray(hc["extrinsic_cv"], np.float64).reshape(3, 4)
        H, W = rgb.shape[:2]
        q = np.asarray(obs["joint_action"]["vector"], np.float32).reshape(-1)
        if not self.hist:
            self.hist = [q] * self.state_hist
        gt = obs.get("object_gt", {})
        clicks, valid = [], []
        for i, nm in enumerate(self.slots):
            o = gt.get(nm)
            uv = project(np.asarray(o["p"], np.float64), K, E) if o is not None else None
            if uv is not None and 0 <= uv[0] < W and 0 <= uv[1] < H:
                clicks.append(tuple(uv)); valid.append(i)
        arm = self.arm_fn(q, K, E, H, W) if self.arm_fn is not None else None
        pr = self.ext(rgb, dep, K, E, clicks, arm_mask=arm)
        objs = {}
        for j, i in enumerate(valid):
            obj = geometry_from_detection(pr.detections[j], dep, K, E, self.geom_src)
            if obj is not None:
                objs[self.slots[i]] = obj
        g = build_slot_obs(depth_to_world(dep, K, E), objs, self.slots, self.n_pts,
                           E=E, K=K, seed=self.t)
        dev = "cuda"
        o = {"patch_tokens": pr.patches[None].to(dev),
             "box2d": torch.from_numpy(g["box2d"])[None].to(dev),
             "xyz": torch.from_numpy(g["xyz"])[None].to(dev),
             "boxgeo": torch.from_numpy(g["boxgeo"])[None].to(dev),
             "pcd_pts": torch.from_numpy(g["pcd_pts"])[None].to(dev),
             "pcd_mask": torch.from_numpy(g["pcd_mask"])[None].to(dev),
             "state": torch.from_numpy(np.concatenate(self.hist))[None].to(dev)}
        self.t += 1
        return self.policy.predict_action(o, generator=self.gen)[0].float().cpu().numpy()


# ---------------- benchmark contract ----------------
def encode_obs(observation):
    return observation


def get_model(usr_args):
    return YamModel(usr_args["ckpt_path"], usr_args["config_path"])


def eval(TASK_ENV, model, observation):  # noqa: A001
    obs = encode_obs(observation)
    model.push_state(obs["joint_action"]["vector"])
    chunk = np.asarray(model.act(obs)).copy()
    n_exec = max(len(chunk) - 1, 1)                     # execute 15/16, blend the seam
    if model._seam is not None and len(chunk) > 1:
        grip = [i for i in (6, 13) if i < chunk.shape[1]]
        g = chunk[0][grip].copy()
        chunk[0] = 0.5 * (model._seam + chunk[0])
        chunk[0][grip] = g
    if len(chunk) > n_exec:
        model._seam = chunk[n_exec].copy()
    for a in chunk[:n_exec]:
        TASK_ENV.take_action(a, action_type="qpos")
        observation = encode_obs(TASK_ENV.get_obs())
        model.push_state(observation["joint_action"]["vector"])


def reset_model(model):
    model.reset()
