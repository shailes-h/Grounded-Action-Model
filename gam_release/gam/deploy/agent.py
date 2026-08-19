"""GAMAgent: language instruction -> Molmo2 planner (object points + memory) -> grounding
(tracked clicks -> WildDet3D boxes / 3D geometry / masked patch tokens, robot silhouette from
the URDF) -> flow-matching policy -> action chunk.

    agent = GAMAgent(AgentConfig(...))
    agent.reset("put the red cup on the tray")
    chunk = agent.act(frame, q)              # (n_exec, action_dim)
    for a in chunk: robot.apply_action(a); agent.push_state(robot.get_state())
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from ..model.policy import GAMPolicy
from ..perception.geometry import (build_slot_obs, cam_to_world_to_E, depth_to_world,
                                   object_from_box, pad_slots)
from ..perception.patch_tokens import PatchTokenExtractor
from ..perception.tracker import PointTracker
from ..planner.client import MockPlanner, PlannerClient
from .interfaces import Frame

ROLE_ORDER = ("pick", "place")            # planner roles -> policy object slots (in order)


@dataclass
class AgentConfig:
    policy_ckpt: str
    wilddet_ckpt: str
    planner_url: str = "http://127.0.0.1:18092"
    tracker_ckpt: Optional[str] = None        # CoTracker3 online checkpoint (optional)
    geometry_source: str = "box2d_depth"      # object 3D geometry from WildDet3D:
                                              #  "box2d_depth" predicted 2D box + depth (axis-aligned; default,
                                              #                closest to the training geometry in our tests)
                                              #  "box3d"       predicted oriented 3D box
    n_points: int = 512
    observe_every: int = 30                   # planner memory update period (steps); 0 = off
    observe_history: int = 2                  # earlier frames sent with each observe (motion cue)
    gripper_open_thresh: float = 0.5          # normalised gripper value above which it counts as open
                                              # (used to detect release events for sub-step progress)
    n_exec: int = 15                          # execute this many steps of each 16-step chunk
    smooth_seam: bool = True                  # blend chunk boundary (grippers excluded)
    gripper_dims: List[int] = field(default_factory=lambda: [6, 13])
    session_prefix: str = "real"
    device: str = "cuda"
    seed: Optional[int] = None


class GAMAgent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.policy = GAMPolicy.from_checkpoint(cfg.policy_ckpt, cfg.device)
        self.n_slots = self.policy.encoder.n_slots
        grid = int(round(self.policy.encoder.n_patch ** 0.5))
        self.patches = PatchTokenExtractor(cfg.wilddet_ckpt, cfg.device,
                                           channel_swap=self.policy.image_channel_swap, grid=grid)
        assert cfg.geometry_source in ("box3d", "box2d_depth"), cfg.geometry_source
        self.planner = (MockPlanner.from_spec(cfg.planner_url) if cfg.planner_url.startswith("mock:")
                        else PlannerClient(cfg.planner_url))
        self.tracker = PointTracker(cfg.tracker_ckpt, cfg.device)
        self.gen = None
        if cfg.seed is not None:
            self.gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
        self._ep = 0
        self.reset("")

    # ---- episode state -----------------------------------------------------------------
    def reset(self, instruction: str):
        self.instruction = instruction
        self.t = 0
        self.hist: List[np.ndarray] = []
        self.objects: List[dict] = []          # planner objects (id, name, role, xy)
        self.slot_of: Dict[int, int] = {}      # tracker index -> policy slot
        self.session_id = None
        self._seam = None
        self.memory_state = ""
        self.progress = ""
        self._frames: List[np.ndarray] = []      # recent frames (subsampled) for observe
        self._releases = 0                       # gripper release events since the last observe
        self._grip_closed = None

    def push_state(self, q: np.ndarray):
        q = np.asarray(q, np.float32).reshape(-1)
        self.hist = [self.hist[-1] if self.hist else q, q]
        # gripper release detection (any gripper going closed -> open) for planner progress
        if self.cfg.gripper_dims:
            g = q[self.cfg.gripper_dims]
            closed = g < self.cfg.gripper_open_thresh
            if self._grip_closed is not None:
                self._releases += int((self._grip_closed & ~closed).sum())
            self._grip_closed = closed

    # ---- planner / grounding -----------------------------------------------------------
    def _plan(self, rgb: np.ndarray):
        self._ep += 1
        self.session_id = f"{self.cfg.session_prefix}_ep{self._ep:04d}_{int(time.time())}"
        r = self.planner.start_session(self.session_id, self.instruction, rgb,
                                       want_place=self.n_slots > 1, want_other=False)
        self.objects = list(r.get("objects", []))
        self.memory_state = r.get("state", "")
        pts = [(o["xy"][0], o["xy"][1]) for o in self.objects]
        self.slot_of = {}
        for role_i, role in enumerate(ROLE_ORDER[:self.n_slots]):
            for i, o in enumerate(self.objects):
                if o["role"] == role:
                    self.slot_of[i] = role_i; break
        self.tracker.start(rgb, pts)
        print(f"[agent] plan: {[(o['name'], o['role'], [round(v) for v in o['xy']]) for o in self.objects]}"
              f" | {self.memory_state}", flush=True)

    def _observe(self, rgb: np.ndarray):
        tracked = [{"id": o["id"], "xy": [float(x), float(y)],
                    "visible": bool(self.tracker.vis[i]) if i < len(self.tracker.vis) else True}
                   for i, (o, (x, y)) in enumerate(zip(self.objects, self.tracker.pts))]
        hist = self._frames[:-1][-self.cfg.observe_history:] if self.cfg.observe_history else None
        r = self.planner.observe(self.session_id, self.t, rgb, tracked, history=hist,
                                 events={"releases": self._releases})
        self._releases = 0
        self.memory_state = r.get("state", "")
        self.progress = r.get("progress", "")
        re_ = {}
        for it in r.get("reacquired", []):
            for i, o in enumerate(self.objects):
                if o["id"] == it["id"]:
                    re_[i] = it["xy"]
        if re_:
            self.tracker.reseed(rgb, re_)
        if r.get("new_targets"):
            # sub-step changed: track the new pick/place entities and route them to the slots
            self.objects = list(r.get("objects", self.objects))
            self.tracker.start(rgb, [(o["xy"][0], o["xy"][1]) for o in self.objects])
            self.slot_of = {}
            for role_i, role in enumerate(ROLE_ORDER[:self.n_slots]):
                for i, o in enumerate(self.objects):
                    if o["role"] == role:
                        self.slot_of[i] = role_i          # entities are appended in time order
        print(f"[agent] memory@{self.t}: {self.memory_state} | PROGRESS: {self.progress} | "
              f"facts={r.get('facts')}", flush=True)

    def _slot_clicks(self):
        clicks = [None] * self.n_slots
        for i, s in self.slot_of.items():
            if i < len(self.tracker.pts):
                clicks[s] = self.tracker.pts[i]
        return clicks

    # ---- one control step --------------------------------------------------------------
    @torch.no_grad()
    def act(self, frame: Frame, q: np.ndarray, arm_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """One control step. arm_mask: (H,W) bool robot silhouette (URDF projection,
        Robot.arm_mask); the checkpoint expects it in the patch mask."""
        rgb, depth = frame.rgb, np.asarray(frame.depth_m, np.float32)
        K = np.asarray(frame.K, np.float64)
        E = cam_to_world_to_E(frame.T_cam2base)
        H, W = rgb.shape[:2]
        q = np.asarray(q, np.float32).reshape(-1)
        if not self.hist:
            self.hist = [q, q]
        if self.cfg.observe_history and self.cfg.observe_every > 0:
            stride = max(1, self.cfg.observe_every // (self.cfg.observe_history + 1))
            if self.t % stride == 0:
                self._frames = (self._frames + [rgb])[-(self.cfg.observe_history + 1):]
        if self.t == 0 or self.session_id is None:
            self._plan(rgb)
        else:
            self.tracker.push(rgb)
            if self.cfg.observe_every > 0 and self.t % self.cfg.observe_every == 0:
                try:
                    self._observe(rgb)
                except Exception as e:  # noqa: BLE001
                    print(f"[agent] planner observe failed: {e}", flush=True)
        clicks = self._slot_clicks()

        # WildDet3D on the tracked clicks: patch tokens + a 2D/3D box per object
        valid_clicks = [c for c in clicks if c is not None]
        pr = self.patches(rgb, depth, K, E, valid_clicks, arm_mask=arm_mask)
        objects: Dict[str, dict] = {}
        di = 0
        for s, c in enumerate(clicks):
            if c is None:
                continue
            det = pr.detections[di]; di += 1
            obj = None
            if self.cfg.geometry_source == "box2d_depth" and det.box2d[2] > det.box2d[0]:
                obj = object_from_box(det.box2d, depth, K, E)
            if obj is None and det.center is not None:
                obj = {"p": det.center, "extents": det.extents, "R": det.R}
            if obj is not None:
                objects[f"slot{s}"] = obj
        pcd = depth_to_world(depth, K, E)
        d = build_slot_obs(pcd, objects, [f"slot{s}" for s in range(self.n_slots)],
                           self.cfg.n_points, E=E, K=K, seed=self.t)
        d = pad_slots(d, self.n_slots)
        tw, th = self.policy.train_image_wh                 # boxes in the training pixel space
        d["box2d"] = d["box2d"] * np.array([tw / W, th / H, tw / W, th / H], np.float32)
        dev = self.cfg.device
        obs = {
            "patch_tokens": pr.patches[None].to(dev),
            "box2d": torch.from_numpy(d["box2d"])[None].to(dev),
            "xyz": torch.from_numpy(d["xyz"])[None].to(dev),
            "boxgeo": torch.from_numpy(d["boxgeo"])[None].to(dev),
            "pcd_pts": torch.from_numpy(d["pcd_pts"])[None].to(dev),
            "pcd_mask": torch.from_numpy(d["pcd_mask"])[None].to(dev),
            "state": torch.from_numpy(np.concatenate(self.hist))[None].to(dev),
        }
        chunk = self.policy.predict_action(obs, generator=self.gen)[0].float().cpu().numpy()
        self.t += 1
        self.last = dict(clicks=clicks, boxes=[d.box2d for d in pr.detections], objects=objects,
                         patch_mask=pr.mask16, detections=pr.detections, arm_mask=arm_mask)
        # chunk boundary smoothing: first step blended with the previous chunk's next step
        n = min(self.cfg.n_exec, len(chunk))
        if self.cfg.smooth_seam and n < len(chunk):
            if self._seam is not None:
                g = chunk[0][self.cfg.gripper_dims].copy()
                chunk[0] = 0.5 * (self._seam + chunk[0])
                chunk[0][self.cfg.gripper_dims] = g
            self._seam = chunk[n].copy()
        return chunk[:n]
