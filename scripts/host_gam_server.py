#!/usr/bin/env python3
"""GAM policy inference server for bi-YAM pick-and-place.

Loads a trained GAM checkpoint and serves actions over HTTP.  Each request
contains one RGB frame, sensor depth, joint state, camera intrinsics &
extrinsics, per-slot click coordinates, and (optionally) joint state history.
The server runs WildDet3D + geometry perception on-the-fly, then feeds
the resulting observation into the flow-matching policy.

===============================================================================
ACTION SPACE
===============================================================================

14-D absolute joint targets:
    [left_joint1 .. left_joint7, right_joint1 .. right_joint7]

The policy was trained with flow-matching on joint-level demonstrations.
Actions are ABSOLUTE joint positions, NOT deltas — command them directly.

===============================================================================
WIRE PROTOCOL
===============================================================================

Both directions are JSON (numpy arrays encoded via json_numpy).

    GET  /act       -> health/introspection
    GET  /healthz   -> liveness

    POST /act       -> action inference

      request body:
        {
          "rgb":         ndarray(H, W, 3) uint8 RGB,         # required
          "depth":       ndarray(H, W) float32, metres,      # required (real sensor depth)
          "state":       ndarray(14,) float32,                # required, current joint angles
          "state_hist":  ndarray(2, 14) float32,              # optional, [t-1, t] states
                                                              # (if omitted, t is duplicated)
          "K":           ndarray(3, 3) float32,               # required, camera intrinsics
          "E":           ndarray(3, 4) or (4, 4) float32,     # required, cam2base extrinsic
          "clicks":      [[u, v], ...] one per slot,         # required, pixel coords on RGB
          "instruction": str,                                 # optional, language instruction
          "timestamp":   float,                               # optional, echoed back
          "debug_viz":   bool,                                # optional, return overlay image
        }

      response body:
        {
          "actions":     ndarray(16, 14) float32,  # ABSOLUTE joint targets
          "dt_ms":       float,                    # server-side inference latency
          "timestamp":   float,                    # echoed from request if provided
          "debug_viz":   str,                      # base64 JPEG overlay (if requested)
        }

Run:
    python scripts/host_gam_server.py \\
        --checkpoint runs/pnp_v5/latest.pt \\
        --config configs/pnp_v5.yaml \\
        --host 0.0.0.0 --port 8204 --track-clicks      # (scripts/run_gam_v5_server.sh)

===============================================================================
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gam.server")

WS = Path(__file__).resolve().parent.parent

DEFAULT_PORT = 8204


def _ensure_paths():
    """Put gam_release on sys.path."""
    src = WS / "gam_release"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


class GAMServer:
    """Loads checkpoint + WildDet3D, runs perception + policy inference."""

    def __init__(
        self,
        checkpoint: Path,
        config: dict,
        device: str = "cuda",
        track_clicks: bool = False,
    ):
        _ensure_paths()

        import torch
        from gam.model.policy import GAMPolicy
        from gam.perception.patch_tokens import PatchTokenExtractor
        from gam.perception.geometry import build_slot_obs, depth_to_world

        self._torch = torch
        self._build_slot_obs = build_slot_obs
        self._depth_to_world = depth_to_world

        # ---- load policy ----
        log.info("Loading GAM policy from %s", checkpoint)
        self.policy = GAMPolicy.from_checkpoint(str(checkpoint), device=device)
        self.policy.eval()
        self.device = device

        self.cfg = config
        self.slots = config["data"]["slots"]
        self.n_slots = len(self.slots)
        self.action_dim = self.policy.action_dim
        self.horizon = self.policy.horizon
        self.n_action_steps = self.policy.n_action_steps
        self.state_hist = self.policy.meta.get("state_hist", 2)
        self.n_points = int(config["perception"].get("n_points", 512))
        self.sensor_depth = bool(config["perception"].get("sensor_depth", False))
        self.geom_src = config["perception"].get("geometry_source", "box2d_depth")
        self.z_band = float(config["perception"].get("z_band", 0.15))

        # ---- load detector ----
        ckpt_path = str(WS / config["perception"]["wilddet_ckpt"])
        grid = int(config["perception"].get("patch_grid", 24))
        bf16 = bool(config["perception"].get("bf16", True))
        ch_swap = bool(config["perception"].get("image_channel_swap", False))
        cell_overlap = float(config["perception"].get("cell_min_overlap", 0.30))

        log.info("Loading WildDet3D from %s (grid=%d)", ckpt_path, grid)
        self.extractor = PatchTokenExtractor(
            checkpoint=ckpt_path, device=device, channel_swap=ch_swap,
            grid=grid, bf16=bf16, cell_min_overlap=cell_overlap,
        )

        # ---- CoTracker3 online click tracking ----
        self.tracker = None
        self._episode_started = False
        self._tracker_frame = 0
        if track_clicks:
            try:
                from gam.perception.tracker import PointTracker
                log.info("Loading CoTracker3 online for click tracking ...")
                self.tracker = PointTracker(device=device)  # no checkpoint yet
                # Load via torch.hub (same as precompute_cotracker_clicks.py)
                model = torch.hub.load(
                    "facebookresearch/co-tracker", "cotracker3_online",
                    trust_repo=True,
                )
                self.tracker.p = model.to(device).eval()
                self.tracker._torch = torch
                log.info("CoTracker3 online loaded — click tracking ENABLED")
            except Exception:
                log.exception("Failed to load CoTracker — click tracking DISABLED")
                self.tracker = None

        # ---- robot arm mask ----
        self.arm_mask_fn = self._build_arm_fn()

        self._lock = threading.Lock()
        log.info(
            "Ready: slots=%s action_dim=%d horizon=%d n_action_steps=%d state_hist=%d "
            "track_clicks=%s",
            self.slots, self.action_dim, self.horizon, self.n_action_steps, self.state_hist,
            self.tracker is not None,
        )

    def _build_arm_fn(self):
        r = self.cfg["perception"].get("robot")
        if not r:
            log.warning("No robot config — arm silhouette disabled")
            return None
        from gam.perception.arm_render import URDFArmMask
        m = URDFArmMask(
            str(WS / r["urdf"]), r["joint_names"],
            T_base_root=np.asarray(r["T_base_root"], np.float64),
            link_filter=r.get("link_filter"),
            max_faces=int(r.get("max_faces", 1500)),
            joint_affine=r.get("joint_affine"),
            joint_mirror=r.get("joint_mirror"),
        )
        log.info("URDF silhouette: %d geometries, %d faces", len(m._nodes), m.n_faces)
        return m

    def reset_tracker(self):
        """Reset the click tracker for a new episode."""
        self._episode_started = False
        self._tracker_frame = 0
        log.info("Tracker reset for new episode")

    def predict(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        state: np.ndarray,
        state_hist,
        K: np.ndarray,
        E: np.ndarray,
        clicks: list,
        reset: bool = False,
        extra_frames: list[np.ndarray] | None = None,
    ) -> tuple:
        """Run perception + policy. Returns (actions, tracked_clicks).
        tracked_clicks is the list of click positions actually used (may differ
        from input clicks when CoTracker tracking is active).

        extra_frames: optional list of intermediate RGB frames captured between
            the previous /act call and this one.  They are pushed into CoTracker
            (without running full perception) so the tracker sees dense video
            and can follow objects reliably."""
        import torch
        import cv2
        from gam.perception.geometry import object_from_box, depth_to_world_grid

        H, W = rgb.shape[:2]
        K = np.asarray(K, np.float64).reshape(3, 3)
        E_raw = np.asarray(E, np.float64)
        # Client sends cam2base (4x4 or 3x4). Convert to base2cam (3x4)
        # which is what the perception pipeline expects (same as episodes.py inversion
        # when extrinsic_is_cam2base=True).
        if E_raw.shape == (4, 4):
            R_c2b = E_raw[:3, :3]
            t_c2b = E_raw[:3, 3]
        else:
            M = E_raw.reshape(3, 4)
            R_c2b = M[:, :3]
            t_c2b = M[:, 3]
        E34 = np.zeros((3, 4), np.float64)
        E34[:, :3] = R_c2b.T
        E34[:, 3] = -R_c2b.T @ t_c2b

        S = self.n_slots
        n_pts = self.n_points

        # ---- CoTracker click tracking ----
        if self.tracker is not None:
            if reset or not self._episode_started:
                # Seed tracker with initial clicks on first frame
                seed_pts = [(float(c[0]), float(c[1])) for c in clicks]
                self.tracker.start(rgb, seed_pts)
                self._episode_started = True
                self._tracker_frame = 0
                log.info("CoTracker seeded with %d clicks on frame 0", len(seed_pts))
            else:
                # Push intermediate frames first so the tracker sees dense
                # video (matching training-time precompute at 15fps).
                if extra_frames:
                    for ef in extra_frames:
                        self.tracker.push(ef)
                        self._tracker_frame += 1
                    log.info("CoTracker: pushed %d extra frames (total %d)",
                             len(extra_frames), self._tracker_frame)
                # Push the current frame and get tracked positions
                tracked_pts = self.tracker.push(rgb, force=True)
                clicks = [[x, y] for x, y in tracked_pts]
                self._tracker_frame += 1
                if self._tracker_frame % 10 == 0:
                    vis = self.tracker.vis
                    log.info("CoTracker frame %d: clicks=%s vis=%s",
                             self._tracker_frame,
                             ["(%.0f,%.0f)" % (x, y) for x, y in tracked_pts],
                             vis)

        # Prepare clicks as pixel tuples (using tracked clicks if available)
        tracked_clicks = clicks  # save for response
        valid = []
        click_px = []
        for i, c in enumerate(clicks):
            u, v = float(c[0]), float(c[1])
            if 0 <= u < W and 0 <= v < H and np.isfinite(u) and np.isfinite(v):
                valid.append(i)
                click_px.append((u, v))

        # Arm mask
        arm = None
        if self.arm_mask_fn is not None:
            log.info("arm_mask joints: %s", np.array2string(state[:14], precision=3, separator=', '))
            arm = self.arm_mask_fn(state[:14], K, E34, H, W)

        # Sensor depth handling
        if self.sensor_depth:
            KD, dscale = self._depth_K(K, rgb.shape, depth_m.shape)
            dep_geo = depth_m.copy()
            if arm is not None:
                am_s = cv2.resize(arm.astype(np.uint8), (dep_geo.shape[1], dep_geo.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
                dep_geo[am_s] = 0.0
            # Table plane removal
            P0, ok0 = depth_to_world_grid(dep_geo, KD, E34)
            hz, ez = np.histogram(P0[ok0][:, 2], bins=200, range=(-0.3, 1.0))
            table_z = float(ez[hz.argmax()] + (ez[1] - ez[0]) / 2)
            Pg, okg = depth_to_world_grid(dep_geo, KD, E34)
            dep_geo[okg & (Pg[..., 2] < table_z + 0.008)] = 0.0
            det_depth = np.zeros((H, W), np.float32)  # detector gets zero depth
        else:
            KD, dscale, dep_geo, det_depth = K, 1.0, depth_m, depth_m

        # Run detector
        pr = self.extractor(rgb, det_depth, K, E34, click_px, arm_mask=arm)

        # Build slot observations
        from gam.perception.geometry import build_slot_obs
        import inspect
        _obfb_kw = {}
        if "z_band" in inspect.signature(object_from_box).parameters:
            _obfb_kw["z_band"] = self.z_band

        objs = {}
        for j, i in enumerate(valid):
            det = pr.detections[j]
            if self.sensor_depth and det.box2d[2] > det.box2d[0]:
                import dataclasses as _dc
                det = _dc.replace(det, box2d=det.box2d * dscale)
            obj = self._geometry_from_det(det, dep_geo, KD, E34, **_obfb_kw)
            if obj is not None:
                objs[self.slots[i]] = obj

        pcd = self._depth_to_world(dep_geo, KD, E34)
        g = self._build_slot_obs(pcd, objs, self.slots, n_pts, E=E34, K=K, seed=0)

        patches = pr.patches.numpy().astype(np.float32)  # (G*G, 1024)

        # State history
        if state_hist is not None:
            full_state = state_hist.astype(np.float32).flatten()  # (state_hist * 14,)
        else:
            # Duplicate current state for history
            full_state = np.tile(state[:14].astype(np.float32), self.state_hist)

        # Build observation dict
        obs = {
            "patch_tokens": torch.from_numpy(patches).unsqueeze(0).to(self.device),
            "box2d": torch.from_numpy(g["box2d"]).unsqueeze(0).to(self.device),
            "xyz": torch.from_numpy(g["xyz"]).unsqueeze(0).to(self.device),
            "boxgeo": torch.from_numpy(g["boxgeo"]).unsqueeze(0).to(self.device),
            "pcd_pts": torch.from_numpy(g["pcd_pts"].astype(np.float32)).unsqueeze(0).to(self.device),
            "pcd_mask": torch.from_numpy(g["pcd_mask"]).unsqueeze(0).to(self.device),
            "state": torch.from_numpy(full_state).unsqueeze(0).to(self.device),
        }

        with self._lock, torch.inference_mode():
            actions = self.policy.predict_action(obs)

        actions = actions[0].cpu().numpy()  # (n_action_steps, action_dim)

        debug_info = {
            "detections": [pr.detections[j] for j in range(len(valid))],
            "valid": valid,
            "click_px": click_px,
            "arm_mask": arm,
            "objs": objs,
            "dep_geo": dep_geo,
            "dep_raw": depth_m,
            "KD": KD,
            "E34": E34,
            "K": K,
            "rgb": rgb,
        }
        return actions, tracked_clicks, debug_info

    def render_debug_viz(self, rgb, actions, state, debug_info, clicks):
        """Render debug overlay: WildDet3D 2D bounding boxes on clicked objects."""
        vis = rgb.copy()
        H, W = vis.shape[:2]

        slot_colors = [
            (255, 80, 80),    # pick_1 — red
            (80, 80, 255),    # pick_2 — blue
            (80, 255, 80),    # place_1 — green
        ]

        for j, i in enumerate(debug_info["valid"]):
            det = debug_info["detections"][j]
            col = slot_colors[i % len(slot_colors)]
            label = self.slots[i] if i < len(self.slots) else f"slot_{i}"
            b = det.box2d.astype(int)
            if b[2] > b[0]:
                cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), col, 2, cv2.LINE_AA)
                cv2.putText(vis, label, (b[0], b[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        return vis

    def _geometry_from_det(self, det, depth_m, K, E, **kwargs):
        from gam.perception.geometry import object_from_box
        obj = None
        if self.geom_src == "box3d" and det.center is not None:
            obj = {"p": det.center, "extents": det.extents, "R": det.R}
        if obj is None and det.box2d[2] > det.box2d[0]:
            obj = object_from_box(det.box2d, depth_m, K, E, **kwargs)
        if obj is None and det.center is not None:
            obj = {"p": det.center, "extents": det.extents, "R": det.R}
        return obj

    @staticmethod
    def _depth_K(K, rgb_shape, depth_shape):
        if tuple(depth_shape[:2]) == tuple(rgb_shape[:2]):
            return K, 1.0
        sy = depth_shape[0] / float(rgb_shape[0])
        sx = depth_shape[1] / float(rgb_shape[1])
        KD = K.copy()
        KD[0, 0] *= sx; KD[1, 1] *= sy
        KD[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        KD[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        return KD, sx


def build_app(server: GAMServer, checkpoint: Path, config_path: str):
    import json_numpy
    json_numpy.patch()
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response

    app = FastAPI(title="GAM Bimanual-YAM Policy Server", version="0.1.0")

    # Debug viz state
    viz_state = {"frame_idx": 0, "episode_idx": 0, "video_writer": None}
    viz_dir = Path("runs/pnp_v5/debug_viz")
    viz_dir.mkdir(parents=True, exist_ok=True)

    def _close_video():
        """Finalize the current episode's mp4."""
        vw = viz_state.get("video_writer")
        if vw is not None:
            vw.release()
            viz_state["video_writer"] = None
            log.info("Saved debug viz video for episode %d", viz_state["episode_idx"])

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "checkpoint": str(checkpoint),
            "config": config_path,
            "slots": server.slots,
            "action_dim": server.action_dim,
            "action_space": "ABSOLUTE joint targets (NOT deltas)",
            "action_layout": "[left_j1..left_j7, right_j1..right_j7]",
            "horizon": server.horizon,
            "n_action_steps": server.n_action_steps,
            "state_hist": server.state_hist,
            "sensor_depth": server.sensor_depth,
            "device": server.device,
            "track_clicks": server.tracker is not None,
        })

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/reset")
    async def reset_tracker() -> JSONResponse:
        """Reset CoTracker state for a new episode."""
        server.reset_tracker()
        _close_video()  # finalize previous episode's mp4
        viz_state["episode_idx"] += 1
        viz_state["frame_idx"] = 0
        log.info("Reset for episode %d", viz_state["episode_idx"])
        return JSONResponse({
            "status": "ok",
            "tracking": server.tracker is not None,
            "debug_viz_dir": str(viz_dir),
        })

    @app.post("/act")
    async def act(request: Request) -> Response:
        import base64
        raw = await request.body()
        log.info("POST /act body size: %.1f KB", len(raw) / 1024.0)
        try:
            payload = json_numpy.loads(raw.decode("utf-8"))
        except Exception as e:
            return _error_response(400, f"failed to decode json_numpy body: {e}")

        try:
            # --- RGB: accept JPEG-compressed (rgb_jpg) or raw numpy (rgb) ---
            if "rgb_jpg" in payload:
                jpg_bytes = base64.b64decode(payload["rgb_jpg"])
                rgb = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            else:
                rgb = np.asarray(payload["rgb"], dtype=np.uint8)

            # --- Depth: accept PNG-compressed uint16 mm (depth_png) or raw float32 metres (depth) ---
            if "depth_png" in payload:
                png_bytes = base64.b64decode(payload["depth_png"])
                depth_mm = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                depth = depth_mm.astype(np.float32) / 1000.0
            else:
                depth = np.asarray(payload["depth"], dtype=np.float32)

            state = np.asarray(payload["state"], dtype=np.float32).flatten()
            K = np.asarray(payload["K"], dtype=np.float64)
            E = np.asarray(payload["E"], dtype=np.float64)
            clicks = payload["clicks"]
        except KeyError as e:
            return _error_response(400, f"missing required field: {e}")
        except Exception as e:
            return _error_response(400, str(e))

        state_hist = payload.get("state_hist")
        if state_hist is not None:
            state_hist = np.asarray(state_hist, dtype=np.float32)
        reset = bool(payload.get("reset", False))

        if state.shape[0] != 14:
            return _error_response(400, f"state must be 14-D, got {state.shape}")
        if len(clicks) != server.n_slots:
            return _error_response(400,
                f"need {server.n_slots} clicks (one per slot), got {len(clicks)}")

        want_viz = bool(payload.get("debug_viz", False))

        # Decode extra intermediate frames for CoTracker dense tracking
        extra_frames = None
        if "extra_frames_jpg" in payload:
            extra_frames = []
            for jpg_b64 in payload["extra_frames_jpg"]:
                jpg_bytes = base64.b64decode(jpg_b64)
                ef = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
                extra_frames.append(cv2.cvtColor(ef, cv2.COLOR_BGR2RGB))

        t0 = time.perf_counter()
        try:
            actions, tracked_clicks, debug_info = server.predict(
                rgb=rgb, depth_m=depth, state=state,
                state_hist=state_hist, K=K, E=E, clicks=clicks,
                reset=reset, extra_frames=extra_frames,
            )
        except Exception as e:
            log.exception("inference failed")
            return _error_response(500, f"inference failed: {e}")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        out: dict[str, Any] = {
            "actions": actions,
            "dt_ms": dt_ms,
            "tracked_clicks": tracked_clicks,
        }
        if "timestamp" in payload:
            out["timestamp"] = payload["timestamp"]

        if want_viz:
            try:
                viz_img = server.render_debug_viz(rgb, actions, state, debug_info, clicks)
                viz_bgr = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", viz_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                out["debug_viz"] = base64.b64encode(buf.tobytes()).decode("ascii")

                # Write frame to mp4 video
                H_v, W_v = viz_bgr.shape[:2]
                if viz_state["video_writer"] is None:
                    vpath = viz_dir / f"episode_{viz_state['episode_idx']:04d}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    viz_state["video_writer"] = cv2.VideoWriter(
                        str(vpath), fourcc, 5.0, (W_v, H_v))
                    log.info("Started video: %s", vpath)
                viz_state["video_writer"].write(viz_bgr)
                viz_state["frame_idx"] += 1
            except Exception:
                log.exception("debug viz render failed")

        return Response(content=json_numpy.dumps(out), media_type="application/json")

    return app


def _error_response(status: int, message: str):
    import json, json_numpy
    from fastapi.responses import Response
    return Response(
        content=json_numpy.dumps({"error": message}),
        status_code=status,
        media_type="application/json",
    )


def warmup(server: GAMServer) -> None:
    """One dummy inference so CUDA kernels are compiled."""
    log.info("Warming up with dummy frame ...")
    H, W = 720, 1280
    dummy_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    dummy_depth = np.ones((H, W), dtype=np.float32) * 0.5
    dummy_state = np.zeros(14, dtype=np.float32)
    # Guessed intrinsics matching the config
    dummy_K = np.array([[896, 0, 640], [0, 896, 360], [0, 0, 1]], dtype=np.float64)
    # cam2base extrinsic from the dataset
    dummy_E = np.array([
        [-0.9894, 0.0612, -0.1321, 0.0089428],
        [0.1283, 0.7949, -0.5930, -0.0306607],
        [0.0687, -0.6036, -0.7943, 0.4099219],
    ], dtype=np.float64)
    dummy_clicks = [[640, 360]] * server.n_slots

    t0 = time.perf_counter()
    try:
        actions, _, _ = server.predict(
            rgb=dummy_rgb, depth_m=dummy_depth, state=dummy_state,
            state_hist=None, K=dummy_K, E=dummy_E, clicks=dummy_clicks,
        )
        log.info("Warmup OK (%.1f ms), actions shape %s",
                 (time.perf_counter() - t0) * 1000.0, actions.shape)
    except Exception:
        log.exception("Warmup failed (server will still start)")


def parse_args():
    p = argparse.ArgumentParser(description="GAM Bimanual-YAM policy inference server")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="path to trained .pt checkpoint (e.g. runs/pnp/latest.pt)")
    p.add_argument("--config", type=str, required=True,
                   help="path to training YAML config (e.g. configs/pnp.yaml)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--track-clicks", action="store_true",
                   help="Enable CoTracker3 online click tracking across frames "
                        "(matches training-time CoTracker precomputation)")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    server = GAMServer(
        checkpoint=args.checkpoint,
        config=config,
        device=args.device,
        track_clicks=args.track_clicks,
    )

    if not args.no_warmup:
        warmup(server)

    app = build_app(server, args.checkpoint, args.config)

    import uvicorn
    log.info("Listening on %s:%d  (POST /act, POST /reset)", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
