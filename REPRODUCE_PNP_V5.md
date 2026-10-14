# Reproducing PnP v5 on the real bi-YAM rig

How to rebuild the PnP v5 pick-and-place policy from scratch on a new machine: data, clicks,
cache, checkpoint and robot server. Every step here ran end-to-end in September 2026
(SLURM jobs 63696 pipeline, 63699 train, 6370x/6373x/63741 robot server). Details for each
tool are in the README sections it links to.

```
LeRobot v3.0          HDF5 / episode         + clicks            cache (.npz)        checkpoint            robot
PnP_version5  ──1──▶  data/pnp_v5_hdf5 ──2,3──▶ clicks/cotracker ──4──▶ data/pnp_v5_cache ──5──▶ runs/pnp_v5 ──6──▶ /act server
```

| | |
| --- | --- |
| task | 4 variants of "Pick the <right top/bottom> yellow mint and <left top/bottom> blue mint in the white box." |
| data | 102 episodes, 37,818 frames @ 15 fps, 1280×720 RGB + D435 depth (aligned to color), 14-D joint state/action |
| slots | `[pick_1, pick_2, place_1]`, sequential clicks (`labeling/annotations/`, committed) |
| config | `configs/pnp_v5.yaml` |
| result | 302.4M params, 12,000 steps, global batch 256 (2 GPUs), loss 1.33 → 0.0032 |

## What is NOT in git (back these up before deleting the machine)

| artifact | size | where it lived | how to get it back |
| --- | --- | --- | --- |
| raw dataset `datasets_GAM/PnP_version5` | 7.9 GB | `~/GAM_train_ready/datasets_GAM/` | our own recording (LeRobot v3.0, `bi_yam_follower`). No public copy, so **back it up**. |
| WildDet3D v1.1 `gam_release/weights/wilddet3d_v1.1/wilddet3d_v1.1_droid_finetuned.ckpt` | 4.5 GB | `gam_release/weights/` | HF snapshot `5eb63fcffef32a2e27637c4850091ddce26ff79d`, sha256 `7e68fb2e4256667bd6942795a549793ae06c3f312ef2fd22db2e791e6189527b` (see README → weights) |
| trained policy `runs/pnp_v5/latest.pt` (+ `step_{3,6,9,12}000.pt`) | 1.2 GB each | `runs/pnp_v5/` | retrain (step 5, ~4 h on 2 GPUs) or back up `latest.pt` |
| converted HDF5 `data/pnp_v5_hdf5` | 24 GB | `data/` | regenerate (steps 1 + 3) |
| cache `data/pnp_v5_cache` | 9.5 GB | `data/` | regenerate (step 4) |

Everything else is committed: code, config, calibration (`configs/calib/bi_yam_d435_cam2base.npy`),
hand clicks (`labeling/annotations/episode_*.json`, 102 files), the URDF + meshes
(`assets/robots/bi_yam/`) and the environment lock (`env/gam_pip_freeze.txt`).

## 0. Environment

```bash
git clone -b yam-shailesh-fork git@github.com:GehaoZhang6/Grounding-Action-Model.git ~/GAM_train_ready
cd ~/GAM_train_ready
conda create -n gam python=3.10 -y && conda activate gam
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r env/gam_pip_freeze.txt        # exact versions that ran v5 (or install as needed)
pip install -e gam_release
export PYTHONPATH=gam_release:$PYTHONPATH
# system: ffmpeg (AV1 decode for the LeRobot videos)
```

You need internet on first use: CoTracker3 comes from `torch.hub` (`facebookresearch/co-tracker`),
and WildDet3D's LingBot-Depth backbone config from the HF hub. Then place the WildDet3D v1.1
checkpoint at the path above and the dataset at `datasets_GAM/PnP_version5`.

**SLURM launchers** (`scripts/run_pnp_v5_*.sh`, `scripts/run_gam_v5_server.sh`) assume the repo is at
`~/GAM_train_ready`, the conda env is at `~/miniconda3` and named `gam`, and they exclude
crane0–2. Edit the `cd`/`conda`/`#SBATCH` lines on another cluster, and submit from the repo root.

## 1. Convert LeRobot → HDF5 (CPU)

```bash
python scripts/convert_lerobot_to_hdf5.py \
    --dataset-dir datasets_GAM/PnP_version5 --out-dir data/pnp_v5_hdf5 \
    --extrinsic-npy configs/calib/bi_yam_d435_cam2base.npy \
    --fx 907.90 --fy 905.86 --cx 643.41 --cy 373.91
```

Check: `data/pnp_v5_hdf5/episode_0000000.hdf5` has `vision/cam_head/colors (458,720,1280,3)`
and `depth (458,720,1280) uint16`.

## 2. Clicks (already done: committed)

`labeling/annotations/episode_XXXXXXX.json` holds three frame-0 clicks per episode, in the order
`pick 1`, `pick 2`, `place`. To redo or extend them (e.g. new episodes):

```bash
python labeling/annotator.py --config configs/pnp_v5.yaml   # -> http://127.0.0.1:8870, saves to labeling/annotations/
```

## 3. Track clicks with CoTracker3 (1 GPU, ~1.5 h)

```bash
sbatch scripts/run_pnp_v5_cotracker.sh
```

This writes `clicks/cotracker` (T,3,2) into every HDF5. The launcher runs on a copy of the config with
`clicks.source: annotations`, because the seeds are the hand clicks.

## 4. Precompute perception cache (1 GPU, ~5.5 h) + visual checks

```bash
sbatch scripts/run_pnp_v5_precompute.sh     # or run_pnp_v5_pipeline.sh = steps 3 + 4 + checks
sbatch scripts/run_pnp_v5_viz.sh            # verify.py videos, viz_cache PNGs, PCD alignment -> runs/pnp_v5/viz/
```

Look at `runs/pnp_v5/viz/` before training:

- boxes sit tight on the two mints and the white box;
- clicks follow the mints when they are lifted;
- the arm silhouette is dimmed in;
- `pcd_*.png` top panel shows depth aligned with the RGB.

## 5. Train (2 GPUs, ~4 h at 1.2 s/it)

```bash
sbatch scripts/run_pnp_v5_train.sh      # torchrun --nproc_per_node=2 train/train.py --config configs/pnp_v5.yaml
```

Expect at start: `[train] 94 episodes / 35167 windows (val held out: [episode_0000094 … 0000101]) world=2`
and `params 302.4M`. Reference loss: 1.33 @ 50 → 0.17 @ 200 → 0.0032 @ 12000.
Optional open-loop check (not run for v5; it went straight to the robot):
`python eval/eval_offline.py --config configs/pnp_v5.yaml --ckpt runs/pnp_v5/latest.pt`.

## Known issues / gotchas

- HDF5s converted before the `_load_tasks` fix store `instruction` as the task index (0–3), not
  the sentence. This is irrelevant while `use_language: false`. Re-convert before turning language on.
- `scripts/inspect_robot_mask.py` / `tune_bimanual_urdf.py` / `optimize_bimanual_urdf.py` need a
  display. Use `tune_extrinsic_web.py` (browser) on headless nodes.
- CoTracker can drift when an object is fully occluded. Spot-check with
  `precompute_cotracker_clicks.py --dump-viz`.
