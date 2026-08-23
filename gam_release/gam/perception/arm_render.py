"""Robot silhouette by projecting the robot's own meshes (URDF) into the camera.

Two layers:
    MeshProjector   — you provide per-link meshes + per-link poses (from your FK) -> mask
    URDFArmMask     — loads a URDF with yourdfpy/trimesh, runs FK from the joint vector -> mask

The silhouette is rasterised triangle by triangle (cv2.fillPoly over all triangles at once),
so it hugs the real robot shape (no convex hulls). This is how the robot mask the policy was
trained with is built — use the same projection at train and test time.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


class MeshProjector:
    """meshes: {link_name: [(vertices (N,3) in link frame, faces (M,3) int), ...]}"""

    def __init__(self, meshes: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]):
        self.meshes = {k: [(np.asarray(v, np.float64), np.asarray(f, np.int64)) for v, f in lst]
                       for k, lst in meshes.items()}

    @staticmethod
    def _draw(out, verts_world, faces, K, E, W, H):
        Pc = verts_world @ E[:, :3].T + E[:, 3]
        z = Pc[:, 2]
        uv = np.stack([K[0, 0] * Pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2],
                       K[1, 1] * Pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]], 1)
        tri_ok = (z > 0.03)[faces].all(1)
        if not tri_ok.any():
            return
        tris = uv[faces[tri_ok]]                                     # (M,3,2)
        inb = ~((tris[:, :, 0].max(1) < 0) | (tris[:, :, 0].min(1) >= W)
                | (tris[:, :, 1].max(1) < 0) | (tris[:, :, 1].min(1) >= H))
        if not inb.any():
            return
        tris = np.clip(tris[inb], -4000, 4000)
        # cv2.fillPoly fills several polygons with the even-odd rule, so overlapping triangles
        # (front and back faces of the same solid) would cancel out. Front-facing and
        # back-facing triangles of one mesh do not overlap among themselves -> fill the two
        # groups separately and OR them.
        e1 = tris[:, 1] - tris[:, 0]; e2 = tris[:, 2] - tris[:, 0]
        front = (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) >= 0
        for sel in (front, ~front):
            if sel.any():
                tmp = np.zeros_like(out)
                cv2.fillPoly(tmp, list(tris[sel].astype(np.int32)), 1)
                np.maximum(out, tmp, out=out)

    @staticmethod
    def _finish(out: np.ndarray, dilate: int, close: int) -> np.ndarray:
        if close:   # fill hairline gaps between rasterised triangles
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
        if dilate:
            out = cv2.dilate(out, np.ones((dilate, dilate), np.uint8))
        return out.astype(bool)

    def render(self, link_poses: Dict[str, np.ndarray], K, E, H: int, W: int,
               dilate: int = 3, close: int = 5) -> np.ndarray:
        """link_poses: {link_name: 4x4 pose of the link in the robot base frame};
        K (3,3); E (3,4) base->camera. Returns (H,W) bool."""
        K = np.asarray(K, np.float64); E = np.asarray(E, np.float64).reshape(3, 4)
        out = np.zeros((H, W), np.uint8)
        for name, T in link_poses.items():
            for v, f in self.meshes.get(name, []):
                T = np.asarray(T, np.float64)
                self._draw(out, v @ T[:3, :3].T + T[:3, 3], f, K, E, W, H)
        return self._finish(out, dilate, close)


class URDFArmMask:
    """URDF -> silhouette. Requires `pip install yourdfpy trimesh`.

    Args:
        urdf_path: robot URDF (mesh paths resolved relative to it / package dirs).
        joint_names: order of the actuated joints in your state vector (extra entries such as
        joint_affine: {joint: [offset, scale]} — applied as q = offset + value*scale before
            rendering, then clipped to the URDF joint limits. Use it when the recorded state
            stores a normalized value (e.g. gripper 0..1) instead of the joint's native unit.
        joint_mirror: {follower: leader} — the follower joint copies the leader's (mapped)
            value; for parallel grippers whose second finger is not in the state vector.
            gripper values are ignored unless they name a URDF joint).
        T_base_root: 4x4 pose of the URDF root link in the policy's base frame (identity if
            your base frame IS the URDF root).
        collision: use collision meshes instead of visual meshes.
        max_faces: meshes with more triangles are decimated (silhouettes need few faces).
        link_filter: only render geometry attached to links whose name contains one of these
            substrings (e.g. ["fl_", "fr_"] for the two arms); None = whole robot.
    """

    def __init__(self, urdf_path: str, joint_names: Sequence[str], T_base_root=None,
                 collision: bool = False, max_faces: int = 1500,
                 joint_affine=None, joint_mirror=None,
                 link_filter: Optional[Sequence[str]] = None):
        import yourdfpy
        self.robot = yourdfpy.URDF.load(urdf_path, build_scene_graph=True, load_meshes=True,
                                        load_collision_meshes=collision, build_collision_scene_graph=collision)
        self.scene = self.robot.collision_scene if collision else self.robot.scene
        self.joint_affine = {k: (float(v[0]), float(v[1])) for k, v in (joint_affine or {}).items()}
        self.joint_mirror = dict(joint_mirror or {})
        self._limits = {}
        self.joint_names = list(joint_names)
        self.T_base_root = np.eye(4) if T_base_root is None else np.asarray(T_base_root, np.float64)
        # geometry nodes to render (optionally restricted to some links), decimated once
        self._nodes = []
        for node in self.scene.graph.nodes_geometry:
            _, gname = self.scene.graph.get(node)
            link = self.scene.graph.transforms.parents.get(node, node)
            if link_filter and not any(sub in link or sub in node for sub in link_filter):
                continue
            m = self.scene.geometry[gname]
            if not hasattr(m, "faces"):
                continue
            if len(m.faces) > max_faces:
                try:                                    # trimesh>=4: keyword face_count (needs fast_simplification)
                    m = m.simplify_quadric_decimation(face_count=max_faces)
                except TypeError:
                    try:
                        m = m.simplify_quadric_decimation(max_faces)
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    pass
            self._nodes.append((node, np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)))
        self.n_faces = int(sum(len(f) for _, _, f in self._nodes))
        for j in self.robot.robot.joints:
            if j.limit is not None and j.limit.lower is not None:
                self._limits[j.name] = (float(j.limit.lower), float(j.limit.upper))

    def link_meshes_world(self, q: np.ndarray):
        cfg = {n: float(v) for n, v in zip(self.joint_names, np.asarray(q).reshape(-1))
               if n in self.robot.actuated_joint_names}
        for n, (off, sc) in (self.joint_affine or {}).items():
            if n in cfg:
                cfg[n] = off + cfg[n] * sc
                lim = self._limits.get(n)
                if lim is not None:
                    cfg[n] = float(np.clip(cfg[n], lim[0], lim[1]))
        for fol, led in (self.joint_mirror or {}).items():
            if led in cfg and fol in self.robot.actuated_joint_names:
                cfg[fol] = cfg[led]
        self.robot.update_cfg(cfg)
        out = []
        for node, v, f in self._nodes:
            T = self.T_base_root @ np.asarray(self.scene.graph.get(node)[0], np.float64)
            out.append((v @ T[:3, :3].T + T[:3, 3], f))
        return out

    def __call__(self, q: np.ndarray, K, E, H: int, W: int, dilate: int = 3,
                 close: int = 5) -> np.ndarray:
        """q: joint vector (see joint_names); K (3,3); E (3,4) base->camera -> (H,W) bool."""
        K = np.asarray(K, np.float64); E = np.asarray(E, np.float64).reshape(3, 4)
        out = np.zeros((H, W), np.uint8)
        for v, f in self.link_meshes_world(q):
            MeshProjector._draw(out, v, f, K, E, W, H)
        return MeshProjector._finish(out, dilate, close)
