"""Online point tracking (CoTracker3 online) for planner-selected object pixels.

Falls back to holding the last known pixels when CoTracker is not installed; the planner's
periodic /observe re-points lost objects in either mode.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


class PointTracker:
    def __init__(self, checkpoint: Optional[str] = None, device: str = "cuda",
                 track_wh: Tuple[int, int] = (640, 360), window_len: int = 16):
        self.dev = device
        self.track_wh = track_wh
        self.pts: List[Tuple[float, float]] = []
        self.vis: List[bool] = []
        self.p = None
        self.buf, self.n = [], 0
        if checkpoint:
            try:
                from cotracker.predictor import CoTrackerOnlinePredictor
                import torch
                self.p = CoTrackerOnlinePredictor(checkpoint=checkpoint,
                                                  window_len=window_len).to(device).eval()
                self._torch = torch
            except Exception as e:  # noqa: BLE001
                print(f"[tracker] CoTracker unavailable ({e}); holding planner points instead")
                self.p = None

    @property
    def available(self) -> bool:
        return self.p is not None

    def _frame(self, rgb: np.ndarray):
        small = cv2.resize(rgb, self.track_wh, interpolation=cv2.INTER_AREA)
        return self._torch.from_numpy(small).permute(2, 0, 1).float()

    def start(self, rgb: np.ndarray, pts: Sequence[Tuple[float, float]]):
        H, W = rgb.shape[:2]
        self.pts = [(float(x), float(y)) for x, y in pts]
        self.vis = [True] * len(self.pts)
        self.W, self.H = W, H
        if self.p is None or not self.pts:
            return
        f = self._frame(rgb)
        q = self._torch.tensor([[[0.0, x * self.track_wh[0] / W, y * self.track_wh[1] / H]
                                 for x, y in self.pts]], device=self.dev)
        self.p(video_chunk=f[None, None].to(self.dev), is_first_step=True, queries=q)
        self.buf, self.n = [f], 1

    def push(self, rgb: np.ndarray, force: bool = False):
        """Push a new frame and return updated point positions.

        Args:
            rgb: new RGB frame (H, W, 3) uint8.
            force: if True, pad the buffer to 16 frames (by repeating the new
                   frame) so that CoTracker runs inference immediately instead
                   of waiting for 16 natural frames.  Use this when the server
                   receives frames sparsely (e.g. once per action chunk) rather
                   than at video rate.
        """
        if self.p is None or not self.pts:
            return self.pts
        f = self._frame(rgb)
        self.buf.append(f); self.n += 1
        self.buf = self.buf[-32:]
        if force:
            if self.n < 16:
                # Pad buffer to 16 by repeating frames so CoTracker can
                # run inference right away instead of waiting for 16 real frames.
                pad_needed = 16 - len(self.buf)
                self.buf = [self.buf[0]] * pad_needed + self.buf
                self.n = 16
            elif (self.n - 16) % self.p.step != 0:
                # Align n to the next step boundary so every forced push
                # triggers inference (frames arrive sparsely, ~1/s).
                self.n = 16 + ((self.n - 16 + self.p.step - 1) // self.p.step) * self.p.step
        if self.n >= 16 and (self.n - 16) % self.p.step == 0:
            chunk = self._torch.stack(self.buf[-16:], 0)[None].to(self.dev)
            with self._torch.no_grad():
                tracks, vis = self.p(video_chunk=chunk)
            self.pts = [(float(x) * self.W / self.track_wh[0], float(y) * self.H / self.track_wh[1])
                        for x, y in tracks[0, -1].tolist()]
            self.vis = [bool(v) for v in (vis[0, -1] > 0.5).tolist()]
        return self.pts

    def reseed(self, rgb: np.ndarray, new_pts: dict):
        """new_pts: {index: (u,v)} — overwrite and restart tracking on the current frame."""
        for i, xy in new_pts.items():
            if 0 <= i < len(self.pts):
                self.pts[i] = (float(xy[0]), float(xy[1])); self.vis[i] = True
        if self.p is not None:
            self.start(rgb, self.pts)
