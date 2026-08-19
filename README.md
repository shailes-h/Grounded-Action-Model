# YAM — train the GAM policy on your dual-arm data

A minimal, self-contained pipeline for the **G**rounded **A**ction **M**odel on a dual-arm
robot: `precompute` turns raw episodes into the exact features the policy consumes
(masked RGB-D patch tokens, per-object boxes/point clouds, robot silhouettes),
`train` runs flow-matching imitation learning, `eval` measures the result offline and
plugs it into a closed-loop harness. Training and evaluation share one perception code
path by construction.

```
raw episodes (HDF5)                       cache (.npz / episode)                 checkpoint
 RGB-D + calib + state/action  ──precompute──▶ patches, boxes, point clouds ──train──▶ latest.pt ──eval──▶ MAE / rollouts
 + one pixel click per object                  state/action windows
```

## Layout

| folder | what |
| --- | --- |
| `configs/` | one YAML per task: your data schema + robot + hyperparameters (`example.yaml` documents every key) |
| `common/` | config loader, episode reader (fully driven by the YAML key map — adapt the config, not the code) |
| `precompute/` | `precompute.py` (episodes → cache), `verify.py` (overlay video — **watch it before training**) |
| `train/` | `dataset.py`, `train.py` (flow matching, EMA, release-format checkpoints) |
| `eval/` | `eval_offline.py` (open-loop MAE + trajectory plots), `deploy_policy.py` (closed-loop contract) |
| `scripts/` | thin bash wrappers |

## 0. Install

Needs the GAM package (model + perception) and its vendored WildDet3D:

```bash
conda create -n gam python=3.11 -y && conda activate gam
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ./gam_release                 # the GAM package (vendored in this branch)
pip install h5py pyyaml matplotlib
```

Weights: the WildDet3D checkpoint (`wilddet3d.pt`, same file the GAM release uses) — path
goes into the config. Nothing else is downloaded.

## 1. Describe your data (`configs/my_task.yaml`)

Copy `configs/example.yaml` and point the `data:` keys at your HDF5 layout. Requirements:

* **head camera**: RGB (`(T,H,W,3)` uint8 or JPEG bytes) + **aligned metric depth** +
  intrinsics + base→camera extrinsics (or the 4×4 camera pose, set `extrinsic_is_cam2base`).
  All 3D the policy sees lives in the robot base frame — calibration accuracy matters.
* **state / action**: `(T, 14)` for two 6-DoF arms + grippers
  (`[arm_L(6), grip_L, arm_R(6), grip_R]`), absolute joint targets. Other layouts work —
  dims are read from the data — but keep state and action in the same layout.
* **one pixel click per object slot** (`data.slots`, manipulated object first). Two sources:
  `project_centers` (you log 3D object centres, e.g. in sim — projected every frame) or
  `dataset` (recorded clicks: `(S,2)` frame-0, held constant, or `(T,S,2)` per frame).
* **robot section**: URDF + the joint-name order of your state vector. The robot silhouette
  is projected into every frame and kept in the patch mask — the policy is trained *with*
  it, so evaluation must provide it too (`deploy_policy` does, from the same config).

## 2. Precompute

```bash
python precompute/precompute.py --config configs/my_task.yaml
# shardable: --ep-start / --ep-end per job; finished episodes are skipped
python precompute/verify.py --config configs/my_task.yaml --episode episode_0000000
```

Two knobs worth knowing (in `perception:`): `patch_grid` — patch tokens are grid×grid
(16→256 tokens, 24→576; larger grid = finer visual detail, ~2.2× more condition tokens at
24); `geometry_source` — the per-object point cloud / 3D box comes either from the depth
inside WildDet3D's 2D box (`box2d_depth`, axis-aligned, default) or from WildDet3D's
predicted oriented 3D box (`box3d`). Both are baked into the cache and the checkpoint;
change ⇒ re-run precompute.

`verify.py` writes an overlay video — check before burning GPU hours:
blue = WildDet3D 2D box (tight on the object), red = 3D box re-projected (on the object,
else calibration/depth is wrong), green = click, dimmed = area the policy never sees,
robot visible inside the kept area. Cost ≈ 0.5 s × frames × slots on one GPU.

Cache contents per episode: `patches (T,256,1024) fp16`, `box2d/xyz/boxgeo`,
`pcd_pts (T,S,512,3)`, `pcd_mask`, `state`, `action`, `clicks`.

## 3. Train

```bash
python train/train.py --config configs/my_task.yaml
```

* ~300 M-parameter multi-stream transformer; flow-matching v-prediction
  (t ∼ 0.999·Beta(1, 1.5)), AdamW 2e-4 (β 0.9/0.95, wd 1e-3; trunk biases / norms /
  positional embeddings not decayed, encoder fully decayed — the reference recipe),
  1 k warmup + cosine, EMA (power 0.75) — the EMA weights are what is saved.
* Regularisation (all on by default): `state_dropout 0.2` zeroes the whole proprioception
  token (forces visual grounding), `det_xyz_dropout 0.1` zeroes the object-centre channel,
  `attn_dropout 0.1` on the transformer attention.
* Reference budget: 50 episodes × ~80 frames trained well at 25 k steps, batch 16, one GPU
  (~4 h). `latest.pt` is release-format: loadable by `GAMPolicy.from_checkpoint`, the GAM
  repo's `smoke_test.py`, and everything under `eval/`.

## 4. Evaluate

Offline (no simulator/robot; uses the `data.val_episodes` held-out tail):

```bash
python eval/eval_offline.py --config configs/my_task.yaml --ckpt runs/my_task/latest.pt
```

Prints per-dimension open-loop MAE and writes recorded-vs-predicted joint plots to
`<out_dir>/eval_offline/`. Sanity bar: arm dims well below the joint range, gripper dims
switching at the right times in the plots.

Closed loop: `eval/deploy_policy.py` implements the common
`get_model / encode_obs / eval / reset_model` contract for dual-arm benchmark harnesses —
pass `{"ckpt_path": ..., "config_path": ...}`; it consumes the harness observation
(head camera + joint vector + object centres for the clicks), runs the *same* perception
as precompute, predicts 16 steps and executes 15 with seam smoothing. For a real robot,
the GAM repository's `gam/deploy/` (Camera/Robot adapters + `run_real.py`) consumes the
same checkpoints.

## Conventions that must match between data, training and deployment

1. base-frame extrinsics (OpenCV camera: +z forward, +y down), metric depth aligned to RGB;
2. state/action layout and units identical everywhere (absolute joint targets);
3. robot silhouette present in the patch mask at train **and** eval;
4. object geometry source (`perception.geometry_source`) and patch grid
   (`perception.patch_grid`) identical at precompute and deployment — the checkpoint stores
   them in its `config`/`meta`, and `deploy_policy` derives both from the checkpoint.

## Troubleshooting

| symptom | check |
| --- | --- |
| red 3D boxes floating off the objects in `verify.py` | extrinsics direction (`extrinsic_is_cam2base`), depth unit, depth-RGB alignment |
| `slots-found 0/S` during precompute | clicks outside the image (projection frame wrong) or `clicks.source` misconfigured |
| robot not dimmed-in in `verify.py` | URDF `joint_names` order vs state vector, `T_base_root`, `link_filter` too narrow |
| loss plateaus high | check `verify.py` first — geometry bugs cap imitation quality |
| eval MAE fine but robot fails closed-loop | harness obs conventions differ from data (depth unit, extrinsics, state layout) |
