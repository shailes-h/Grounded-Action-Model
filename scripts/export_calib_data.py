"""Export RGB + URDF mesh + depth PCD + calibration for the interactive alignment tool.

    python scripts/export_calib_data.py [--config configs/pnp_v5.yaml] [--episode episode_0000000]
        [--out data/calib_data.json]

Writes one JSON (frame 0 of the episode): RGB as base64 JPEG, depth back-projected to camera-
frame points (<=500k, 1 mm precision), URDF mesh vertices in the base frame (<=5k), the stored
cam2base extrinsic, K, frame-0 clicks and joint state. The camera extrinsic
(configs/calib/bi_yam_d435_cam2base.npy) was visually tuned in a browser alignment tool fed by
this JSON (the tool itself is not in this repo): depth points mapped through a candidate
cam2base are overlaid on the URDF mesh until the arms coincide.
"""
import argparse, json, base64, sys, os
import numpy as np
import cv2
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.config import load_config
from common.episodes import list_episodes, load_episode

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/pnp_v5.yaml")
ap.add_argument("--episode", default="episode_0000000")
ap.add_argument("--out", default="data/calib_data.json")
args = ap.parse_args()

cfg = load_config(args.config)
# Allow running before clicks exist (fresh HDF5s)
cfg["data"]["clicks"] = {"source": "language"}
src = [p for p in list_episodes(cfg) if args.episode in p]
ep = load_episode(src[0], cfg)

H, W = ep.rgb(0).shape[:2]
K = ep.K[0].astype(np.float64)

# 1. RGB → JPEG base64
rgb = ep.rgb(0)
_, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
rgb_b64 = base64.b64encode(buf.tobytes()).decode()
print(f"RGB JPEG: {len(rgb_b64)//1024} KB (base64)")

# 2. Depth → camera-frame points (subsampled)
dep = ep.depth_m(0)
us, vs = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
z = dep.astype(np.float64).ravel()
ok = (z > 0.05) & (z < 3.0)
x_c = (us.ravel() - K[0, 2]) / K[0, 0] * z
y_c = (vs.ravel() - K[1, 2]) / K[1, 1] * z
Pc = np.stack([x_c, y_c, z], 1)[ok]
rng = np.random.default_rng(42)
# Keep all points but round to 3 decimals (1mm precision) to shrink JSON
Pc_sub = Pc
# If JSON would exceed ~12MB, subsample to fit
MAX_PTS = 500000
if len(Pc_sub) > MAX_PTS:
    idx = rng.choice(len(Pc_sub), MAX_PTS, replace=False)
    Pc_sub = Pc_sub[idx]
print(f"Depth PCD: {len(Pc)} total → {len(Pc_sub)} subsampled")

# 3. URDF mesh in world frame
from gam.perception.arm_render import URDFArmMask

r = cfg["perception"]["robot"]
arm = URDFArmMask(
    r["urdf"], r["joint_names"],
    T_base_root=np.asarray(r["T_base_root"], np.float64),
    link_filter=r.get("link_filter"), max_faces=400,
    joint_affine=r.get("joint_affine"), joint_mirror=r.get("joint_mirror"),
)
meshes = arm.link_meshes_world(ep.state[0])
all_v, all_f = [], []
off = 0
for v, f in meshes:
    all_v.append(v.astype(np.float64))
    all_f.append(f.astype(np.int64) + off)
    off += len(v)
all_v = np.vstack(all_v)
all_f = np.vstack(all_f)
# Subsample vertices for the browser (dots, not full wireframe)
if len(all_v) > 5000:
    idx_m = rng.choice(len(all_v), 5000, replace=False)
    all_v = all_v[idx_m]
print(f"URDF mesh: {len(all_v)} verts (subsampled for browser)")

# 4. Stored extrinsic (cam2base 4×4)
with h5py.File(src[0], "r") as fh:
    E_stored = np.array(fh["vision/cam_head/extrinsic"], dtype=np.float64)

# 5. Click positions at frame 0
clicks = ep.clicks[0]  # (S, 2)

data = {
    "rgb_b64": rgb_b64,
    "W": W, "H": H,
    "K": K.tolist(),
    "E_cam2base": E_stored.tolist(),
    "depth_cam": np.round(Pc_sub, 3).tolist(),  # 1mm precision
    "mesh_verts": np.round(all_v, 5).tolist(),
    "clicks": np.round(clicks, 1).tolist(),
    "slots": cfg["data"]["slots"],
    "joint_state": ep.state[0].tolist(),
}

out = args.out
with open(out, "w") as f:
    json.dump(data, f)
print(f"\nWrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")
print(f"E_cam2base:\n{E_stored}")
print(f"K:\n{K}")
ep.close()
