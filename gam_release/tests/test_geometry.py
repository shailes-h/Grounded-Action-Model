import numpy as np

from gam.perception.geometry import (backproject, box2d_from_obb, build_slot_obs, cam_to_world_to_E,
                                     crop_obb, depth_to_world, depth_to_world_grid, object_from_box,
                                     pad_slots, project)


def _cam():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    T = np.eye(4); T[:3, 3] = [0.3, 0.0, 0.8]                    # camera 0.8 m above the base origin
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])   # looking straight down
    return K, T, cam_to_world_to_E(T)


def test_project_backproject_roundtrip():
    K, T, E = _cam()
    p = np.array([0.35, 0.1, 0.2])
    uv = project(p, K, E)
    z = (E[:, :3] @ p + E[:, 3])[2]
    p2 = backproject(uv[0], uv[1], z, K, E)
    assert np.allclose(p, p2, atol=1e-6)


def test_depth_to_world_matches_grid():
    K, T, E = _cam()
    d = np.full((480, 640), 0.6, np.float32); d[:10] = 0
    pts = depth_to_world(d, K, E)
    Pw, ok = depth_to_world_grid(d, K, E)
    assert pts.shape == (ok.sum(), 3)
    assert np.allclose(pts, Pw[ok], atol=1e-5)
    assert np.allclose(pts[:, 2], 0.2, atol=1e-5)          # 0.8 - 0.6 above the base plane


def test_object_from_box_and_slot_obs():
    K, T, E = _cam()
    d = np.full((480, 640), 0.8, np.float32)                # table at z=0
    d[200:280, 300:380] = 0.7                               # a 10 cm tall block
    obj = object_from_box([300, 200, 379, 279], d, K, E)
    assert obj is not None and abs(obj["p"][2] - 0.1) < 0.02
    pcd = depth_to_world(d, K, E)
    o = build_slot_obs(pcd, {"a": obj}, ["a", "b"], 128, E=E, K=K)
    assert o["pcd_pts"].shape == (2, 128, 3) and o["pcd_mask"][0].sum() > 0 and o["pcd_mask"][1].sum() == 0
    assert (o["box2d"][0] > 0).all() and (o["box2d"][1] == 0).all()
    assert np.allclose(o["boxgeo"][0, 3:], [1, 0, 0, 0, 1, 0])
    o3 = pad_slots(o, 3)
    assert o3["xyz"].shape == (3, 3) and o3["pcd_pts"].shape == (3, 128, 3)


def test_box2d_from_obb_and_crop():
    K, T, E = _cam()
    b = box2d_from_obb([0.3, 0.0, 0.1], [0.1, 0.1, 0.1], np.eye(3), E, K)
    assert b[2] > b[0] and b[3] > b[1]
    pts = np.random.RandomState(0).uniform(-0.2, 0.2, (5000, 3)) + [0.3, 0.0, 0.1]
    out, m = crop_obb(pts, [0.3, 0.0, 0.1], [0.1, 0.1, 0.1], np.eye(3), 64)
    assert m.sum() == 64 and (np.abs(out[m] - [0.3, 0.0, 0.1]) <= 0.05 * 1.05 + 0.01 + 1e-6).all()
