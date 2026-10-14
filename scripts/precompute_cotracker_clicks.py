#!/usr/bin/env python3
"""Track the frame-0 clicks across the whole episode with CoTracker3 (streamed online --
offline OOMs past a few hundred frames, see the note further down), and write
the result back into the episode's HDF5 so `data.clicks.source: dataset` can consume it
as a `(T,S,2)` per-frame array -- fixing the "clicks held constant" limitation of the
`annotations`/`project_centers` sources (a picked-up object's click stays where the
object USED to be; see precompute/verify.py to see this drift for yourself).

Mimics origin/gam-libero's two-phase design (gam/data/precompute_cotracker_500.py +
precompute_phase3_clickfused.py), collapsed into one script -- but using CoTracker3
ONLINE (sliding window) rather than gam-libero's OFFLINE call, because our episodes
(~1000 frames) are much longer than gam-libero's (~125 frames/demo) and offline mode
holds every frame's features in memory at once (confirmed OOM on a 992-frame episode,
31GB GPU, single episode, batch 1):

    python scripts/precompute_cotracker_clicks.py --config configs/my_task.yaml --episode episode_0000000
    # seeds come from the config's clicks -> run it with data.clicks.source: annotations

Writes `clicks/cotracker` (T,S,2) float32 and `clicks/cotracker_vis` (T,S) bool into the
HDF5 file itself. Then point your config at it:

    data:
      clicks: {source: dataset, dataset_key: clicks/cotracker}

WARNING: CoTracker tracks PIXELS, not identity -- if the object is briefly fully occluded
(inside a container, behind the gripper) the track can drift onto whatever's visible at
that pixel and not recover. `pred_vis` (saved alongside) flags CoTracker's own visibility
estimate per frame; frames where it's False are exactly where you should NOT trust the
track blindly -- inspect them (see --dump-viz) before committing to this over the static
click for a given episode.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import h5py
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from common.config import load_config          # noqa: E402
from common.episodes import list_episodes, load_episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--episode", default="", help="one episode name; default = all episodes.glob matches")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump-viz", default="", help="dir to write a quick overlay video per episode "
                                                     "(green=visible, red=CoTracker flags occluded)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = list_episodes(cfg)
    if args.episode:
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] == args.episode]
        assert paths, f"no episode named {args.episode}"

    # cotracker3_offline holds every frame's features in memory at once -- fine for
    # gam-libero's ~125-frame demos, OOMs on our longer (~1000 frame) episodes. Use the
    # ONLINE variant instead (same one gam/perception/tracker.py already wraps for
    # deploy-time tracking): sliding window, streamed frame by frame.
    print("[cotrack] loading CoTracker3 online (torch.hub, facebookresearch/co-tracker) ...")
    cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online", trust_repo=True).to(args.device).eval()
    window_len = cot.step * 2

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        ep = load_episode(path, cfg)
        S = len(cfg["data"]["slots"])
        q0 = ep.clicks[0]                     # (S,2) -- held-constant sources give the same
        assert np.isfinite(q0).all(), f"{name}: frame-0 clicks have NaNs, fix annotations first"
        T = ep.T

        frames = np.stack([ep.rgb(t) for t in range(T)])       # (T,H,W,3) uint8
        H, W = frames.shape[1:3]
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(args.device)  # (T,3,H,W)
        queries = torch.tensor([[0.0, float(x), float(y)] for x, y in q0],
                                dtype=torch.float32, device=args.device)[None]  # (1,S,3)

        print(f"[cotrack] {name}: T={T} S={S} window={window_len} -- tracking (streamed) ...")
        tracks_full = np.zeros((T, S, 2), np.float32)
        vis_full = np.zeros((T, S), bool)
        tracks_full[0] = q0
        vis_full[0] = True
        # match gam/perception/tracker.py's proven call pattern EXACTLY: grow a real
        # frame buffer one frame at a time (capped at window_len), only invoke the model
        # every `step` frames on that buffer's last window_len frames. Re-slicing the raw
        # video array in step-sized jumps (the first version of this script) does NOT
        # propagate the predictor's internal temporal state correctly -- verified broken
        # (tracked points didn't move even when the real object visibly did).
        with torch.no_grad():
            cot(video_chunk=video[0:1][None], is_first_step=True, queries=queries)
            buf = [video[0]]
            last_end = 0
            for n in range(1, T):
                buf.append(video[n])
                buf = buf[-window_len:]
                if n >= window_len - 1 and (n - (window_len - 1)) % cot.step == 0:
                    chunk = torch.stack(buf, 0)[None]
                    pred_tracks, pred_vis = cot(video_chunk=chunk)   # (1,window_len,S,2)
                    # this call's output covers frames [n-window_len+1, n]; only the
                    # portion after what we already wrote is new
                    t = pred_tracks.shape[1]
                    win_start = n - t + 1
                    start = max(win_start, last_end)
                    if start <= n:
                        off = start - win_start
                        tracks_full[start:n + 1] = pred_tracks[0, off:].cpu().numpy()
                        vis_full[start:n + 1] = (pred_vis[0, off:] > 0.5).cpu().numpy()
                        last_end = n + 1
            if last_end < T:               # tail shorter than one step -- hold last value
                tracks_full[last_end:] = tracks_full[last_end - 1]
                vis_full[last_end:] = vis_full[last_end - 1]
        tracks, vis = tracks_full, vis_full
        occluded_frac = 1.0 - vis.mean()
        print(f"[cotrack] {name}: done -- {occluded_frac*100:.1f}% of (frame,slot) flagged occluded by CoTracker")

        ep.close()
        with h5py.File(path, "a") as f:
            for k in ("clicks/cotracker", "clicks/cotracker_vis"):
                if k in f:
                    del f[k]
            f.create_dataset("clicks/cotracker", data=tracks)
            f.create_dataset("clicks/cotracker_vis", data=vis)
        print(f"[cotrack] {name}: wrote clicks/cotracker (T,S,2) + clicks/cotracker_vis (T,S) into {path}")

        if args.dump_viz:
            os.makedirs(args.dump_viz, exist_ok=True)
            out_path = os.path.join(args.dump_viz, f"{name}_cotracker.mp4")
            vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
            for t in range(T):
                img = frames[t][:, :, ::-1].copy()
                for s in range(S):
                    x, y = tracks[t, s]
                    color = (0, 255, 0) if vis[t, s] else (0, 0, 255)
                    cv2.circle(img, (int(x), int(y)), 6, color, -1)
                    cv2.putText(img, cfg["data"]["slots"][s], (int(x) + 8, int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                vw.write(img)
            vw.release()
            print(f"[cotrack] {name}: wrote {out_path}")

    print("COTRACK_DONE")


if __name__ == "__main__":
    main()
