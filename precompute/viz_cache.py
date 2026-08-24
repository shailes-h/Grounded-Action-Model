"""Render check images straight from the precomputed cache — no GPU needed.

    python precompute/viz_cache.py --config configs/my_task.yaml --episode episode0 \
        [--frames 0,300,600] [--out viz]

For each requested frame, writes `<out>/<episode>_f<t>.png`:
  left  = the raw camera frame with the per-slot clicks (green cross) drawn on it,
  right = what the policy actually sees — kept patch cells bright (yellow grid),
          masked cells dark, WildDet3D 2D boxes red.

Look for: boxes hugging the objects, the robot silhouette present in the kept cells,
no large background areas kept, clicks on the right objects. If any of that is off,
fix the config (clicks / robot section / cell_min_overlap) and re-run precompute
before training. `precompute/verify.py` renders the same overlay as a full video.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.config import load_config          # noqa: E402
from common.episodes import list_episodes, load_episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--episode", required=True, help="episode name (npz stem)")
    ap.add_argument("--frames", default="", help="comma list; default first/middle/last")
    ap.add_argument("--out", default="viz")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cache = os.path.join(cfg["train"]["cache_dir"], args.episode + ".npz")
    assert os.path.exists(cache), f"no cache {cache} — run precompute first"
    z = np.load(cache, allow_pickle=True)
    path = [p for p in list_episodes(cfg)
            if os.path.splitext(os.path.basename(p))[0] == args.episode]
    assert path, f"episode {args.episode} not found by data.episodes_glob"
    ep = load_episode(path[0], cfg)
    T = len(z["state"])
    G = int(round(z["patches"].shape[1] ** 0.5))
    frames = ([int(v) for v in args.frames.split(",") if v != ""]
              if args.frames else [0, T // 2, T - 1])
    os.makedirs(args.out, exist_ok=True)
    for t in frames:
        t = int(np.clip(t, 0, T - 1))
        img = ep.rgb(t)[:, :, ::-1].copy()                       # BGR for cv2 drawing
        H, W = img.shape[:2]
        left = img.copy()
        for s_i in range(z["box2d"].shape[1]):                   # clicks (green cross)
            uv = ep.clicks[t, s_i]
            if np.isfinite(uv).all():
                cv2.drawMarker(left, (int(uv[0]), int(uv[1])), (0, 255, 0),
                               cv2.MARKER_CROSS, 22, 3)
        kept = (np.abs(z["patches"][t].astype(np.float32)).sum(-1) > 0).reshape(G, G)
        right = (img * 0.3).astype(np.uint8)
        for r in range(G):
            for c in range(G):
                if kept[r, c]:
                    y0, y1 = r * H // G, (r + 1) * H // G
                    x0, x1 = c * W // G, (c + 1) * W // G
                    right[y0:y1, x0:x1] = img[y0:y1, x0:x1]
                    cv2.rectangle(right, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 1)
        for s_i in range(z["box2d"].shape[1]):                   # detected boxes (red)
            b = z["box2d"][t, s_i]
            if b[2] > b[0]:
                cv2.rectangle(right, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              (0, 0, 255), 2)
        row = np.hstack([left, right])
        cv2.putText(row, f"{args.episode} t={t}  kept {kept.mean() * 100:.0f}% of {G}x{G}",
                    (8, 26), 0, 0.8, (0, 255, 255), 2)
        out = os.path.join(args.out, f"{args.episode}_f{t}.png")
        cv2.imwrite(out, row)
        print("wrote", out)
    ep.close()
    print("VIZ_DONE")


if __name__ == "__main__":
    main()
