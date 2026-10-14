#!/usr/bin/env python3
"""Web-based interactive robot-silhouette alignment tuner (headless-friendly).

    python scripts/tune_extrinsic_web.py --config configs/pnp_v5.yaml [--port 8871]
    -> open http://<host>:8871  (port-forward if remote)

Shows the RGB image with URDF mesh vertices and depth PCD projected using the episode's
camera calibration.  Six sliders (tx, ty, tz, rx, ry, rz) nudge
perception.robot.T_base_root (URDF root -> policy base frame); the overlay updates live.
"Save" prints the tuned T_base_root to paste into the config -- it does not write
any file, and the camera extrinsic itself is left untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np
from flask import Flask, Response, jsonify, request
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.config import load_config
from common.episodes import list_episodes, load_episode

app = Flask(__name__)
STATE = {}


def _project(pts_world, K, E_cam2base):
    """Project world-frame 3D points to pixel coords using cam2base extrinsic."""
    R = E_cam2base[:3, :3]
    t = E_cam2base[:3, 3]
    # cam2base means: P_base = R @ P_cam + t  =>  P_cam = R^T @ (P_base - t)
    pts_cam = (R.T @ (pts_world.T - t[:, None])).T  # (N, 3)
    z = pts_cam[:, 2]
    ok = z > 0.01
    uv = np.zeros((len(pts_world), 2), np.float64)
    uv[ok, 0] = K[0, 0] * pts_cam[ok, 0] / z[ok] + K[0, 2]
    uv[ok, 1] = K[1, 1] * pts_cam[ok, 1] / z[ok] + K[1, 2]
    return uv, ok


def _get_frame_base(t):
    """Get RGB and depth PCD overlay (fixed, no URDF) for frame t. Cached."""
    s = STATE
    ep = s["ep"]
    t = max(0, min(t, ep.T - 1))
    if s.get("_base_t") == t:
        return s["_base"]
    rgb = ep.rgb(t)
    K = ep.K[t].astype(np.float64)
    dep = ep.depth_m(t)
    H, W = rgb.shape[:2]
    E_cam = s["E_current"]
    # Depth -> camera -> world -> reproject (cyan PCD overlay, baked in)
    us, vs = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    z = dep.astype(np.float64).ravel()
    ok = (z > 0.05) & (z < 3.0)
    x_c = (us.ravel() - K[0, 2]) / K[0, 0] * z
    y_c = (vs.ravel() - K[1, 2]) / K[1, 1] * z
    Pc = np.stack([x_c, y_c, z], 1)[ok]
    R_e, t_e = E_cam[:3, :3], E_cam[:3, 3]
    Pw = (R_e @ Pc.T + t_e[:, None]).T
    rng = np.random.default_rng(42)
    if len(Pw) > 50000:
        idx = rng.choice(len(Pw), 50000, replace=False)
        Pw = Pw[idx]
    # Project PCD back to image (fixed overlay)
    uv_pcd, ok_pcd = _project(Pw, K, E_cam)
    base = rgb.copy()
    step = max(1, len(uv_pcd) // 8000)
    for u, v, m in zip(uv_pcd[::step, 0], uv_pcd[::step, 1], ok_pcd[::step]):
        if m and 0 <= int(u) < W and 0 <= int(v) < H:
            cv2.circle(base, (int(u), int(v)), 1, (0, 220, 220), -1)
    data = {"base": base, "K": K, "joint_state": ep.state[t]}
    s["_base_t"] = t
    s["_base"] = data
    return data


def _render_overlay(T_base_root, t=0):
    """Render RGB with fixed cyan PCD + adjustable red URDF mesh overlay."""
    s = STATE
    fd = _get_frame_base(t)
    rgb = fd["base"].copy()
    H, W = rgb.shape[:2]
    K = fd["K"]
    E_cam = s["E_current"]

    # Rebuild arm with adjusted T_base_root and get mesh
    s["arm"].T_base_root = T_base_root
    meshes = s["arm"].link_meshes_world(fd["joint_state"])
    all_v = []
    for v, f in meshes:
        all_v.append(v.astype(np.float64))
    all_v = np.vstack(all_v)
    rng = np.random.default_rng(42)
    if len(all_v) > 10000:
        idx_m = rng.choice(len(all_v), 10000, replace=False)
        all_v = all_v[idx_m]

    # Project URDF mesh vertices (red dots) using the fixed cam2base extrinsic
    uv_mesh, ok_mesh = _project(all_v, K, E_cam)
    for u, v, m in zip(uv_mesh[:, 0], uv_mesh[:, 1], ok_mesh):
        if m and 0 <= int(u) < W and 0 <= int(v) < H:
            cv2.circle(rgb, (int(u), int(v)), 2, (255, 50, 50), -1)

    cv2.putText(rgb, f"t={t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _T_from_params(params):
    """Build 4x4 T_base_root from [tx, ty, tz, rx, ry, rz]."""
    tx, ty, tz, rx, ry, rz = params
    R = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def _params_from_T(T):
    """Extract [tx, ty, tz, rx, ry, rz] from 4x4 T_base_root."""
    t = T[:3, 3].tolist()
    r = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True).tolist()
    return t + r


@app.get("/")
def index():
    return PAGE


@app.get("/overlay")
def overlay():
    params = [float(request.args.get(k, 0)) for k in ["tx", "ty", "tz", "rx", "ry", "rz"]]
    t = int(request.args.get("t", 0))
    T_br = _T_from_params(params)
    return Response(_render_overlay(T_br, t), mimetype="image/jpeg")


@app.get("/api/init")
def api_init():
    p = _params_from_T(STATE["T_base_root_init"])
    return jsonify({"params": p,
                    "labels": ["tx", "ty", "tz", "rx", "ry", "rz"],
                    "ranges": [[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05],
                               [-5, 5], [-5, 5], [-5, 5]],
                    "steps": [0.0005, 0.0005, 0.0005, 0.05, 0.05, 0.05],
                    "T": STATE["ep"].T})


@app.post("/api/save")
def api_save():
    d = request.get_json(force=True)
    params = d["params"]
    T_new = _T_from_params(params)

    # Print the new T_base_root for pasting into the config
    print(f"\n{'='*60}")
    print(f"Tuned T_base_root (paste into perception.robot of your config):")
    print(f"    T_base_root: {T_new.tolist()}")
    print(f"{'='*60}\n")

    STATE["T_base_root_init"] = T_new
    STATE["arm"].T_base_root = T_new
    # Invalidate frame cache
    STATE.pop("_base_t", None)

    return jsonify({"ok": True, "T_base_root": T_new.tolist()})


PAGE = """<!doctype html><meta charset=utf-8><title>Extrinsic Tuner</title>
<style>
body{background:#14161a;color:#dde;font:14px system-ui;margin:0;display:flex;flex-direction:column;align-items:center;padding:20px}
h2{margin:8px 0}
#img{border:1px solid #333;max-width:100%}
.slider-row{display:flex;align-items:center;gap:10px;margin:4px 0;width:800px}
.slider-row label{width:30px;text-align:right;font-weight:bold;color:#8cf}
.slider-row input[type=range]{flex:1}
.slider-row .val{width:80px;font-family:monospace;color:#ff8}
button{background:#2a3f5a;color:#dde;border:1px solid #456;border-radius:6px;padding:8px 20px;margin:6px;cursor:pointer;font-size:14px}
button:hover{background:#3a5f8a}
#status{color:#7c7;margin:6px}
</style>
<h2>URDF T_base_root Tuner</h2>
<p style="color:#889">Red dots = URDF mesh (adjustable), Cyan dots = depth PCD (fixed). Align red onto the robot arms.</p>
<img id=img>
<div id=sliders></div>
<div>
  <button onclick="resetSliders()">Reset</button>
  <button onclick="save()" style="background:#1d4d2a;border-color:#3a7">Save (print T_base_root)</button>
</div>
<div id=status></div>
<script>
let params=[], labels=[], ranges=[], steps=[], initParams=[];
let debounce=null, T=1, curFrame=0;

async function boot(){
  const r = await (await fetch('api/init')).json();
  params = r.params;
  initParams = [...r.params];
  labels = r.labels;
  ranges = r.ranges;
  steps = r.steps;
  T = r.T;
  buildSliders();
  refresh();
}

function buildSliders(){
  const div = document.getElementById('sliders');
  div.innerHTML = '';
  // Frame slider first
  const frow = document.createElement('div');
  frow.className = 'slider-row';
  frow.innerHTML = `<label style="color:#fa0">frame</label>
    <input type=range id="slFrame" min="0" max="${T-1}" step="1" value="${curFrame}">
    <span class=val id="vlFrame" style="color:#fa0">${curFrame} / ${T-1}</span>`;
  div.appendChild(frow);
  frow.querySelector('input').oninput = function(){
    curFrame = parseInt(this.value);
    document.getElementById('vlFrame').textContent = curFrame + ' / ' + (T-1);
    clearTimeout(debounce);
    debounce = setTimeout(refresh, 150);
  };
  // Extrinsic sliders
  params.forEach((v, i) => {
    const row = document.createElement('div');
    row.className = 'slider-row';
    const lo = v + ranges[i][0], hi = v + ranges[i][1];
    row.innerHTML = `<label>${labels[i]}</label>
      <input type=range id="sl${i}" min="${lo}" max="${hi}" step="${steps[i]}" value="${v}">
      <span class=val id="vl${i}">${v.toFixed(4)}</span>`;
    div.appendChild(row);
    row.querySelector('input').oninput = function(){
      params[i] = parseFloat(this.value);
      document.getElementById('vl'+i).textContent = params[i].toFixed(4);
      clearTimeout(debounce);
      debounce = setTimeout(refresh, 80);
    };
  });
}

function refresh(){
  const qs = labels.map((l,i) => l+'='+params[i]).join('&');
  document.getElementById('img').src = 'overlay?' + qs + '&t=' + curFrame + '&_=' + Date.now();
}

function resetSliders(){
  params = [...initParams];
  buildSliders();
  refresh();
}

async function save(){
  document.getElementById('status').textContent = 'Saving...';
  const r = await (await fetch('api/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({params})
  })).json();
  if(r.ok){
    document.getElementById('status').innerHTML = 'T_base_root printed to terminal ✓<br><code style="color:#ff8;font-size:12px">' +
      JSON.stringify(r.T_base_root) + '</code>';
  } else {
    document.getElementById('status').textContent = 'Error saving';
  }
}

boot();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, default=8871)
    ap.add_argument("--episode", default="episode_0000000")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # Temporarily bypass clicks requirement
    cfg_load = dict(cfg)
    cfg_load["data"] = dict(cfg["data"])
    cfg_load["data"]["clicks"] = {"source": "language"}

    eps = list_episodes(cfg)
    ep_path = [p for p in eps if args.episode in p][0]
    ep = load_episode(ep_path, cfg_load)

    H, W = ep.rgb(0).shape[:2]

    # Current extrinsic from HDF5
    with h5py.File(ep_path, "r") as fh:
        E_stored = np.array(fh[cfg["data"]["extrinsic_key"]], dtype=np.float64)
    if E_stored.shape == (3, 4):
        E4 = np.eye(4)
        E4[:3, :] = E_stored
        E_stored = E4

    # URDF arm renderer
    from gam.perception.arm_render import URDFArmMask
    r = cfg["perception"]["robot"]
    arm = URDFArmMask(
        r["urdf"], r["joint_names"],
        T_base_root=np.asarray(r["T_base_root"], np.float64),
        link_filter=r.get("link_filter"), max_faces=800,
        joint_affine=r.get("joint_affine"), joint_mirror=r.get("joint_mirror"),
    )

    T_base_root = np.asarray(cfg["perception"]["robot"]["T_base_root"], np.float64)

    STATE["ep"] = ep
    STATE["arm"] = arm
    STATE["E_current"] = E_stored
    STATE["T_base_root_init"] = T_base_root
    STATE["cfg"] = cfg

    print(f"[tuner] RGB {W}x{H}, T={ep.T} frames")
    print(f"[tuner] Current E_cam2base:\n{E_stored}")
    print(f"[tuner] Current T_base_root:\n{T_base_root}")
    print(f"[tuner] open http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
