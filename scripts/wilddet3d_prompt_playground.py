#!/usr/bin/env python
"""Interactive WildDet3D text-prompt playground (RGB-D, websocket).

Loads a single frame straight out of a LeRobot dataset (defaults to episode 0,
frame 0 of datasets_GAM/PnP_version5), builds the WildDet3D detector, and
serves a tiny websocket page: type a text prompt (comma-separated for several
classes at once), hit Detect, and see the 2D boxes drawn on the RGB-D frame.

Reuses as much of the existing pipeline as possible instead of re-implementing
it:
  * frame decoding            -> scripts/convert_lerobot_to_hdf5.py
  * preprocessing constants   -> gam_release/gam/perception/patch_tokens.py
  * detector construction     -> wilddet3d.inference.build_model
  * calibration constants     -> scripts/eval_reference_script.py

Usage:
    python scripts/wilddet3d_prompt_playground.py \\
        --dataset-dir datasets_GAM/PnP_version5 --episode 0 --frame 0 \\
        --checkpoint gam_release/weights/wilddet3d_v1.1/wilddet3d_v1.1_droid_finetuned.ckpt

    -> open http://localhost:8600
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import os
import sys

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "gam_release"))

from gam.perception.patch_tokens import IMEAN, ISTD, INPUT  # noqa: E402

# `scripts/` isn't a package -- pull the frame-decode helpers straight out of
# convert_lerobot_to_hdf5.py by module path instead of duplicating them.
_spec = importlib.util.spec_from_file_location(
    "convert_lerobot_to_hdf5", os.path.join(REPO_ROOT, "scripts", "convert_lerobot_to_hdf5.py"))
_c2h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c2h)

# Same tuned color-camera intrinsics eval_reference_script.py ships with.
INTRINSIC_FX, INTRINSIC_FY, INTRINSIC_CX, INTRINSIC_CY = 907.90, 905.86, 643.41, 373.91

BOX_COLORS = [(66, 135, 245), (245, 96, 66), (66, 245, 134), (245, 215, 66),
              (191, 66, 245), (66, 245, 227)]


def load_frame(dataset_dir: str, episode: int, frame: int, video_key: str = "observation.images.top"):
    """RGB (H,W,3) uint8 + depth (H,W) float32 metres for one dataset frame, via the
    same episode-row / video-decode helpers the HDF5 converter uses."""
    ep_rows = _c2h._episode_rows(dataset_dir)
    ep_row = ep_rows[ep_rows["episode_index"] == episode].iloc[0]
    fps = 15.0
    from_ts = float(ep_row[f"videos/{video_key}/from_timestamp"])
    vid_chunk = int(ep_row[f"videos/{video_key}/chunk_index"])
    vid_file = int(ep_row[f"videos/{video_key}/file_index"])
    video_path = os.path.join(dataset_dir, "videos", video_key, f"chunk-{vid_chunk:03d}", f"file-{vid_file:03d}.mp4")
    frame_idx_in_video = np.round((from_ts + frame / fps) * fps).astype(np.int64)
    rgb = _c2h._decode_video_frames(video_path, np.array([frame_idx_in_video]), fps)[0]

    data_chunk = int(ep_row["data/chunk_index"])
    data_file = int(ep_row["data/file_index"])
    data_path = os.path.join(dataset_dir, "data", f"chunk-{data_chunk:03d}", f"file-{data_file:03d}.parquet")
    import pandas as pd
    df = pd.read_parquet(data_path)
    df = df[df["episode_index"] == episode].reset_index(drop=True)
    depth_col = "observation.images.top_depth"
    if depth_col in df.columns:
        import io
        from PIL import Image
        cell = df[depth_col].iloc[frame]
        png_bytes = cell["bytes"] if isinstance(cell, dict) else cell
        arr = np.asarray(Image.open(io.BytesIO(png_bytes)), np.uint16)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        depth_m = arr.astype(np.float32) / 1000.0
    else:
        depth_m = np.full(rgb.shape[:2], 0.5, np.float32)
    return rgb, depth_m


def preprocess(rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray, device: str):
    """Square-resize to the backbone's input resolution -- same recipe as
    PatchTokenExtractor._forward so results line up with the trained pipeline."""
    H0, W0 = rgb.shape[:2]
    sx, sy = INPUT / W0, INPUT / H0
    im = cv2.resize(rgb, (INPUT, INPUT))
    x = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float()[None] / 255.0
    x = ((x - IMEAN) / ISTD).to(device)
    Ks = torch.from_numpy((np.diag([sx, sy, 1.0]) @ K).astype(np.float32)).to(device)[None]
    dgt = torch.from_numpy(cv2.resize(depth_m, (INPUT, INPUT), interpolation=cv2.INTER_NEAREST)
                            ).to(device)[None, None]
    return x, Ks, dgt, (sx, sy)


def draw_detections(rgb: np.ndarray, boxes, scores, class_ids, texts, sxsy) -> np.ndarray:
    sx, sy = sxsy
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for b, s, c in zip(boxes, scores, class_ids):
        x0, y0, x1, y1 = (b.float().cpu().numpy() / [sx, sy, sx, sy]).astype(int)
        color = BOX_COLORS[int(c) % len(BOX_COLORS)]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        label = f"{texts[int(c)]} {float(s):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x0, max(0, y0 - th - 6)), (x0 + tw + 4, y0), color, -1)
        cv2.putText(out, label, (x0 + 2, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out


def jpeg_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return base64.b64encode(buf).decode("ascii")


PAGE = """<!doctype html><html><head><title>WildDet3D prompt playground</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:20px}
#wrap{max-width:1100px;margin:0 auto}
img{max-width:100%;border:1px solid #333;border-radius:6px;display:block;margin-top:12px}
input[type=text]{width:60%;padding:8px;font-size:14px;border-radius:4px;border:1px solid #444;
  background:#1c1c1c;color:#eee}
button{padding:8px 16px;font-size:14px;border-radius:4px;border:0;background:#3a7afe;color:#fff;
  cursor:pointer;margin-left:8px}
button:disabled{opacity:0.5;cursor:default}
#det{white-space:pre-wrap;background:#1c1c1c;border-radius:6px;padding:10px;margin-top:12px;
  font-family:ui-monospace,monospace;font-size:13px;min-height:1.4em}
#status{color:#888;font-size:13px;margin-top:6px}
</style></head><body><div id="wrap">
<h2>WildDet3D text-prompt playground</h2>
<div>Prompt(s), comma-separated: <input id="prompt" type="text" value="mint tin" placeholder="e.g. mint tin, hand"/>
<button id="go">Detect</button></div>
<div id="status">connecting...</div>
<img id="frame" src=""/>
<div id="det"></div>
<script>
const ws = new WebSocket(`ws://${location.host}/ws`);
const statusEl = document.getElementById('status');
const imgEl = document.getElementById('frame');
const detEl = document.getElementById('det');
const goBtn = document.getElementById('go');
ws.onopen = () => { statusEl.textContent = 'connected — loaded frame ready'; };
ws.onclose = () => { statusEl.textContent = 'disconnected'; goBtn.disabled = true; };
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.image) imgEl.src = 'data:image/jpeg;base64,' + msg.image;
  if (msg.detections !== undefined) {
    detEl.textContent = msg.detections.length
      ? msg.detections.map(d => `${d.text}  score=${d.score.toFixed(3)}  box=[${d.box.map(v=>v.toFixed(0)).join(', ')}]`).join('\\n')
      : '(no detections above threshold)';
    statusEl.textContent = `done in ${msg.dt_ms.toFixed(0)} ms`;
    goBtn.disabled = false;
  }
  if (msg.error) { statusEl.textContent = 'error: ' + msg.error; goBtn.disabled = false; }
};
function send() {
  goBtn.disabled = true;
  statusEl.textContent = 'running...';
  ws.send(JSON.stringify({prompt: document.getElementById('prompt').value}));
}
goBtn.onclick = send;
document.getElementById('prompt').addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
</script></div></body></html>"""


def build_app(rgb, depth_m, K, model, device, score_threshold):
    app = FastAPI()

    @app.get("/")
    async def index():
        return HTMLResponse(PAGE)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        await ws.send_json({"image": jpeg_b64(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))})
        try:
            while True:
                payload = await ws.receive_json()
                texts = [t.strip() for t in payload.get("prompt", "").split(",") if t.strip()]
                if not texts:
                    await ws.send_json({"error": "empty prompt"})
                    continue
                import time
                wall0 = time.time()
                x, Ks, dgt, sxsy = preprocess(rgb, depth_m, K, device)
                with torch.no_grad():
                    boxes, boxes3d, scores, scores2d, scores3d, class_ids, depth_maps = model(
                        images=x, intrinsics=Ks, input_hw=[(INPUT, INPUT)], original_hw=[(INPUT, INPUT)],
                        padding=[(0, 0, 0, 0)], input_texts=texts, depth_gt=dgt)
                dt_ms = (time.time() - wall0) * 1000.0
                b, s, c = boxes[0], scores[0], class_ids[0]
                keep = s >= score_threshold
                b, s, c = b[keep], s[keep], c[keep]
                overlay = draw_detections(rgb, b, s, c, texts, sxsy)
                dets = [{"text": texts[int(ci)], "score": float(si),
                         "box": (bi.float().cpu().numpy() / [sxsy[0], sxsy[1], sxsy[0], sxsy[1]]).tolist()}
                        for bi, si, ci in zip(b, s, c)]
                await ws.send_json({"image": jpeg_b64(overlay), "detections": dets, "dt_ms": dt_ms})
        except WebSocketDisconnect:
            pass

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default="datasets_GAM/PnP_version5")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--video-key", default="observation.images.top")
    ap.add_argument("--checkpoint", default="gam_release/weights/wilddet3d_v1.1/wilddet3d_v1.1_droid_finetuned.ckpt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8600)
    args = ap.parse_args()

    dataset_dir = args.dataset_dir if os.path.isabs(args.dataset_dir) else os.path.join(REPO_ROOT, args.dataset_dir)
    checkpoint = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(REPO_ROOT, args.checkpoint)

    print(f"[playground] loading episode {args.episode} frame {args.frame} from {dataset_dir}")
    rgb, depth_m = load_frame(dataset_dir, args.episode, args.frame, args.video_key)
    K = np.array([[INTRINSIC_FX, 0, INTRINSIC_CX], [0, INTRINSIC_FY, INTRINSIC_CY], [0, 0, 1]], np.float64)

    print(f"[playground] building WildDet3D on {args.device} (this loads a ~4.7GB checkpoint)...")
    from wilddet3d.inference import build_model
    model = build_model(checkpoint=checkpoint, score_threshold=0.0, nms=True, iou_threshold=0.6,
                         device=args.device, skip_pretrained=True, use_depth_input_test=True).eval()

    app = build_app(rgb, depth_m, K, model, args.device, args.score_threshold)
    print(f"[playground] open http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
