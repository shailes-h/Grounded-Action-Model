import re

import numpy as np

from gam.perception.arm_render import MeshProjector
from gam.perception.geometry import cam_to_world_to_E
from gam.planner.client import MockPlanner


def _parse_points():
    # import the parser without pulling torch/transformers (server imports them lazily)
    import importlib
    return importlib.import_module("gam.planner.server").parse_points


def test_molmo2_point_parser_formats():
    pp = _parse_points()
    # observed Molmo2 format: frame index + (id x y) triples, 0-1000 normalised
    assert pp('<points coords="1 1 500 250 2 100 900">a</points>', 1000, 1000) == [(500.0, 250.0), (100.0, 900.0)]
    assert pp('<points coords="1 500 250"/>', 200, 100) == [(100.0, 25.0)]      # (id x y)
    assert pp('<points coords="500 250"/>', 200, 100) == [(100.0, 25.0)]        # (x y)
    assert pp('x="50" y="25"', 200, 100) == [(100.0, 25.0)]                     # legacy percent
    assert pp("not visible", 200, 100) == []


def test_mock_planner_roles():
    m = MockPlanner.from_spec("mock:10,20;30,40;50,60")
    r = m.start_session("s", "put a on b", None, want_place=True)
    assert [o["role"] for o in r["objects"]] == ["pick", "place", "other"]
    r = m.start_session("s", "pick a", None, want_place=False)
    assert [o["role"] for o in r["objects"]] == ["pick", "other"]


def test_mesh_projector_cube_silhouette():
    # unit cube centred 0.5 m in front of a camera looking down -> a filled square in the mask
    K = np.array([[400.0, 0, 160], [0, 400.0, 120], [0, 0, 1]])
    T = np.eye(4); T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]); T[:3, 3] = [0, 0, 1.0]
    E = cam_to_world_to_E(T)
    v = np.array([[x, y, z] for x in (-.05, .05) for y in (-.05, .05) for z in (-.05, .05)])
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    proj = MeshProjector({"cube": [(v, f)]})
    Tl = np.eye(4); Tl[:3, 3] = [0, 0, 0.5]                     # cube 0.5 m below the camera
    m = proj.render({"cube": Tl}, K, E, 240, 320, dilate=0, close=0)
    ys, xs = np.where(m)
    # nearest face is 0.45 m away -> half-width 400*0.05/0.45 = 44.4 px -> ~89 px wide silhouette
    assert m[120, 160] and abs((xs.max() - xs.min()) - 89) <= 3 and abs((ys.max() - ys.min()) - 89) <= 3
    assert not proj.render({"cube": Tl}, K, E, 240, 320)[0, 0]
