#!/usr/bin/env python3
"""Convert a LeRobot v3.0 dataset (parquet + mp4, e.g. datasets_GAM/<name>) into the
per-episode HDF5 layout `common/episodes.py` reads (see configs/example.yaml).

What this DOES map from the source dataset (real data, no guessing):
  * RGB frames        <- videos/<video_key>/chunk-*/file-*.mp4 (ffmpeg, seeking to just
                         this episode's frame span of the shared video)
  * state / action    <- observation.state_joint_angles / action_joint_angles
                         (already `[arm_L(6), grip_L, arm_R(6), grip_R]`, 14-dim -> no
                         reordering needed)
  * instruction       <- meta/tasks.parquet, looked up by task_index
  * depth             <- observation.images.top_depth (uint16 mm PNG bytes in the parquet)
                         when the dataset records it (e.g. PnP_version5, D435
                         aligned-to-color)

What it CANNOT map, because the source dataset does not record it, and fills with
placeholders instead (see the loud warning printed at the end):
  * depth             -> ONLY if there is no `observation.images.top_depth` column:
                         constant plane at `--depth-default-m` (default 0.5 m).
  * intrinsics        -> a guessed pinhole (fx=fy=0.7*W, cx=W/2, cy=H/2) unless you pass
                         --fx/--fy/--cx/--cy.
  * extrinsics        -> identity cam2base (i.e. camera frame == robot base frame) unless
                         you pass --extrinsic-npy (a (4,4) cam2base .npy file).

Point clouds / 3D boxes produced from these placeholders are geometrically meaningless —
this only gets you far enough to smoke-test the precompute/train/eval plumbing (patches,
2D boxes, masks, state/action windows). Get real depth + calibration before trusting any
3D output or training a checkpoint you intend to deploy.

There are also no clicks anywhere in the source (no project_centers, no recorded 2D
clicks). The generated config points `data.clicks.source` at `annotations` — you still
need to hand-click pick/place with `labeling/annotator.py` before precompute will find
any slots.

Usage:
    python scripts/convert_lerobot_to_hdf5.py \\
        --dataset-dir datasets_GAM/PnP_version5 \\
        --out-dir data/pnp_v5_hdf5 \\
        --extrinsic-npy configs/calib/bi_yam_d435_cam2base.npy \\
        --fx 907.90 --fy 905.86 --cx 643.41 --cy 373.91
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import h5py
import numpy as np
import pandas as pd


def _load_tasks(dataset_dir: str) -> dict:
    tasks_path = os.path.join(dataset_dir, "meta", "tasks.parquet")
    df = pd.read_parquet(tasks_path)
    # index is the task string, column 'task_index' is the int id
    return {int(row.task_index): task for task, row in df.iterrows()}


def _episode_rows(dataset_dir: str) -> pd.DataFrame:
    """Concatenate meta/episodes/chunk-*/file-*.parquet -> one row per episode."""
    import glob
    paths = sorted(glob.glob(os.path.join(dataset_dir, "meta", "episodes", "chunk-*", "file-*.parquet")))
    assert paths, f"no meta/episodes/chunk-*/file-*.parquet under {dataset_dir}"
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _probe_wh(video_path: str) -> tuple:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split(",")
    return int(h), int(w)


def _decode_video_frames(video_path: str, frame_indices: np.ndarray, fps: float) -> np.ndarray:
    """Decode only the needed frame range from an mp4 using ffmpeg seek.
    Memory-efficient: only holds frames for one episode (~400 frames, ~1GB).

    Uses input-level seeking (-ss before -i) which is frame-accurate for the
    contiguous ranges LeRobot episodes produce, then decodes exactly the span
    from the first to the last requested frame.
    """
    idx = np.asarray(frame_indices, np.int64)
    if idx.min() < 0:
        raise RuntimeError(f"{video_path}: negative frame index {idx.min()}")
    H, W = _probe_wh(video_path)
    frame_bytes = H * W * 3
    first, last = int(idx.min()), int(idx.max())
    n_span = last - first + 1  # exact number of frames in the contiguous range

    # -ss before -i = input-level seek (fast, keyframe-accurate for AV1)
    # -frames:v = decode exactly n_span frames starting from the seek point
    seek_time = first / fps
    proc = subprocess.run(
        ["ffmpeg", "-v", "error",
         "-ss", f"{seek_time:.6f}",
         "-i", video_path,
         "-frames:v", str(n_span),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True)
    raw = proc.stdout
    n_decoded = len(raw) // frame_bytes
    if n_decoded == 0:
        raise RuntimeError(f"{video_path}: ffmpeg decoded 0 frames from seek={seek_time:.3f}")
    buf = np.frombuffer(raw, np.uint8, count=n_decoded * frame_bytes).reshape(n_decoded, H, W, 3)

    # Map requested indices into the decoded span
    local_indices = idx - first
    if local_indices.max() >= n_decoded:
        raise RuntimeError(f"{video_path}: needed frame offset {local_indices.max()} "
                           f"but only decoded {n_decoded} (seek={seek_time:.3f}, span={n_span})")
    result = buf[local_indices].copy()
    del buf, raw
    return result


def convert_episode(dataset_dir: str, out_dir: str, ep_row, tasks: dict, video_key: str,
                     fps: float, intrinsic: np.ndarray, extrinsic_cam2base: np.ndarray,
                     depth_default_m: float) -> str:
    ep_idx = int(ep_row["episode_index"])
    data_chunk = int(ep_row["data/chunk_index"])
    data_file = int(ep_row["data/file_index"])
    vid_chunk = int(ep_row[f"videos/{video_key}/chunk_index"])
    vid_file = int(ep_row[f"videos/{video_key}/file_index"])
    row_from = int(ep_row["dataset_from_index"])
    row_to = int(ep_row["dataset_to_index"])          # exclusive
    from_ts = float(ep_row[f"videos/{video_key}/from_timestamp"])

    data_path = os.path.join(dataset_dir, "data", f"chunk-{data_chunk:03d}", f"file-{data_file:03d}.parquet")
    video_path = os.path.join(dataset_dir, "videos", video_key, f"chunk-{vid_chunk:03d}", f"file-{vid_file:03d}.mp4")

    df = pd.read_parquet(data_path)
    # Each parquet file may be one episode or contain multiple — filter by episode_index
    if "episode_index" in df.columns:
        df = df[df["episode_index"] == ep_idx].reset_index(drop=True)
    T = len(df)
    assert T > 0, f"episode {ep_idx}: no rows in {data_path}"

    state = np.stack(df["observation.state_joint_angles"].to_numpy()).astype(np.float32)
    action = np.stack(df["action_joint_angles"].to_numpy()).astype(np.float32)
    timestamps = df["timestamp"].to_numpy(np.float64)
    # `timestamp` is EPISODE-relative (resets to 0 at the start of every episode);
    # `from_ts` is this episode's offset into the shared video -- ADD them to get the
    # global video frame index. (Bug fixed: this used to subtract, which is only
    # coincidentally correct for episode 0 where from_ts=0 -- every later episode got a
    # negative index that silently wrapped around to the wrong end of the video.)
    frame_idx_in_video = np.round((from_ts + timestamps) * fps).astype(np.int64)

    rgb = _decode_video_frames(video_path, frame_idx_in_video, fps)
    H, W = rgb.shape[1:3]

    # --- depth: real sensor PNG from parquet, or constant placeholder ---
    depth_col = "observation.images.top_depth"
    if depth_col in df.columns:
        import io
        from PIL import Image
        depth_mm = np.zeros((T, H, W), np.uint16)
        for t_i in range(T):
            cell = df[depth_col].iloc[t_i]
            png_bytes = cell["bytes"] if isinstance(cell, dict) else cell
            img = Image.open(io.BytesIO(png_bytes))
            arr = np.asarray(img, np.uint16)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            depth_mm[t_i] = arr
    else:
        depth_mm = np.full((T, H, W), depth_default_m * 1000.0, np.uint16)

    task_index = int(df["task_index"].iloc[0])
    instruction = tasks.get(task_index, "")

    out_path = os.path.join(out_dir, f"episode_{ep_idx:07d}.hdf5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("vision/cam_head/colors", data=rgb, compression="gzip", compression_opts=4)
        f.create_dataset("vision/cam_head/depth", data=depth_mm, compression="gzip", compression_opts=4)
        f.create_dataset("vision/cam_head/intrinsic", data=intrinsic.astype(np.float32))
        f.create_dataset("vision/cam_head/extrinsic", data=extrinsic_cam2base.astype(np.float32))
        f.create_dataset("joint_state/vector", data=state)
        f.create_dataset("joint_action/vector", data=action)
        f.create_dataset("instruction", data=instruction)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, help="LeRobot dataset root, e.g. datasets_GAM/PnP_version5")
    ap.add_argument("--out-dir", required=True, help="where to write episode_*.hdf5")
    ap.add_argument("--video-key", default=None, help="feature name of the video to use as head cam "
                                                        "(default: auto-detect the only video feature)")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--extrinsic-npy", default=None, help="(4,4) cam2base .npy; default identity")
    ap.add_argument("--depth-default-m", type=float, default=0.5,
                     help="constant placeholder depth plane (metres) written for every pixel")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    info = json.load(open(os.path.join(args.dataset_dir, "meta", "info.json")))
    fps = float(info["fps"])

    video_key = args.video_key
    if video_key is None:
        video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
        assert len(video_keys) == 1, f"multiple/no video features {video_keys}, pass --video-key"
        video_key = video_keys[0]
    vh = info["features"][video_key]["shape"]
    H, W = vh[0], vh[1]

    fx = args.fx if args.fx is not None else 0.7 * W
    fy = args.fy if args.fy is not None else fx
    cx = args.cx if args.cx is not None else W / 2.0
    cy = args.cy if args.cy is not None else H / 2.0
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)

    if args.extrinsic_npy:
        extrinsic_cam2base = np.load(args.extrinsic_npy)
        assert extrinsic_cam2base.shape == (4, 4)
    else:
        extrinsic_cam2base = np.eye(4)

    tasks = _load_tasks(args.dataset_dir)
    episodes = _episode_rows(args.dataset_dir)

    written = []
    for _, ep_row in episodes.iterrows():
        out_path = convert_episode(args.dataset_dir, args.out_dir, ep_row, tasks, video_key,
                                    fps, intrinsic, extrinsic_cam2base, args.depth_default_m)
        print(f"[convert] episode {int(ep_row['episode_index'])} -> {out_path}")
        written.append(out_path)

    print(f"\n[convert] wrote {len(written)} episode(s) to {args.out_dir}")
    print("[convert] WARNING: unless you passed --fx/--fy/--cx/--cy and --extrinsic-npy, "
          "intrinsics/extrinsics are guessed/identity, and without a top_depth column depth is "
          "a constant plane — 3D outputs (xyz, boxgeo, point clouds) are then NOT physically "
          "meaningful. Supply real depth + calibration before training a checkpoint you intend "
          "to deploy.")
    print("[convert] No clicks in the source data — before precompute, hand-click pick/place with:\n"
          f"    python labeling/annotator.py --config <your_config.yaml>   # episodes_glob -> {args.out_dir}/episode_*.hdf5")


if __name__ == "__main__":
    main()
