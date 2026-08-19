"""Template hardware adapters — copy this file, fill in the TODOs for your robot/camera and
pass the class paths to run_real.py:

    python -m gam.deploy.run_real --config configs/deploy_example.yaml \
        --instruction "..." --camera my_adapters.MyCamera --robot my_adapters.MyRobot

Conventions (see README "Conventions"): OpenCV camera axes (+z forward, +y down), depth in
metres aligned to RGB, all poses in the robot BASE frame the checkpoint was trained in.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from gam.deploy.interfaces import Camera, Frame, Robot


class MyCamera(Camera):
    def __init__(self):
        # TODO: open your RGB-D camera (RealSense / ZED / Kinect ...), enable depth-to-color
        # alignment, and read the color intrinsics.
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], np.float32)  # noqa: F821  TODO
        # TODO: hand-eye calibration result: 4x4 pose of the (OpenCV) camera in the robot base
        # frame. If your calibration tool gives base->camera, invert it here.
        self.T_cam2base = np.eye(4, dtype=np.float32)

    def read(self) -> Frame:
        # TODO: grab an aligned RGB-D pair
        rgb = np.zeros((720, 1280, 3), np.uint8)          # (H,W,3) uint8, RGB order
        depth_m = np.zeros((720, 1280), np.float32)       # (H,W) metres, 0 = invalid
        return Frame(rgb=rgb, depth_m=depth_m, K=self.K, T_cam2base=self.T_cam2base,
                     stamp=time.time())


class MyRobot(Robot):
    """State / action layouts MUST match the checkpoint (`policy.meta` printed at start-up),
    e.g. 14-d = [arm_L(6), gripper_L, arm_R(6), gripper_R] absolute joint positions."""

    def __init__(self):
        pass  # TODO: connect to the robot controller

    def get_state(self) -> np.ndarray:
        # TODO: current joint positions (+ gripper) in the training layout / units
        return np.zeros(14, np.float32)

    def apply_action(self, action: np.ndarray) -> None:
        # TODO: command ONE step of absolute joint targets and return after ~1 control period
        # (the run loop paces at --hz). Add your own joint-limit / velocity clamps here.
        pass

    def arm_mask(self, frame: Frame) -> Optional[np.ndarray]:
        # Required: project the robot's own meshes (URDF) at the current joint state into the
        # camera -> (H,W) bool silhouette (kept in the patch mask, as during training).
        #   from gam.perception.arm_render import URDFArmMask
        #   from gam.perception.geometry import cam_to_world_to_E
        #   self.urdf_mask = URDFArmMask("robot.urdf", joint_names=[...], T_base_root=np.eye(4),
        #                                link_filter=["arm_", "gripper_"])   # arm links only
        #   E = cam_to_world_to_E(frame.T_cam2base); H, W = frame.rgb.shape[:2]
        #   return self.urdf_mask(self.get_state(), frame.K, E, H, W)
        return None
