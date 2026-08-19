"""Hardware adapter interfaces. Implement these two classes for your robot + camera."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Frame:
    rgb: np.ndarray            # (H,W,3) uint8, RGB order
    depth_m: np.ndarray        # (H,W) float32 metres, 0 = invalid, aligned to rgb
    K: np.ndarray              # (3,3) intrinsics of the RGB-D image
    T_cam2base: np.ndarray     # (4,4) camera pose in the robot base frame (OpenCV camera axes)
    stamp: float = 0.0


class Camera:
    """RGB-D camera. `read()` must return a depth-aligned frame."""

    def read(self) -> Frame:
        raise NotImplementedError


class Robot:
    """Robot arm(s). State/action layouts must match the checkpoint's training data."""

    def get_state(self) -> np.ndarray:
        """Proprioceptive state vector (e.g. joint positions + gripper), shape (state_dim,)."""
        raise NotImplementedError

    def apply_action(self, action: np.ndarray) -> None:
        """Execute ONE action step (shape (action_dim,)) and block for one control period."""
        raise NotImplementedError

    def arm_mask(self, frame: Frame) -> Optional[np.ndarray]:
        """(H,W) bool silhouette of the robot in the camera image: the robot's URDF meshes
        projected at the current joint state (`gam.perception.arm_render.URDFArmMask`).
        The policy was trained with this silhouette kept in the patch mask — required."""
        return None
