# GAM — Grounded Action Model with a Molmo2 planner (real-robot deployment)

Language instruction → **Molmo2 high-level planner** (points at the task objects, keeps an
episode memory) → **grounding with WildDet3D** (tracked object pixels → predicted 2D/3D boxes →
object geometry + masked RGB-D patch tokens; robot silhouette projected from the URDF) →
**flow-matching action policy** (multi-stream MMDiT, ~300M) → 16-step joint-position chunks.

```
instruction ─┐
             ▼
   ┌───────────────────┐   pick / place / other points        ┌──────────────────────┐
   │ Molmo2 planner    │ ───────────────────────────────────▶ │ point tracker        │
   │ (gam.planner)     │ ◀─── tracked pixels every N steps ── │ (CoTracker, optional)│
   │  + episode memory │                                       └──────────┬───────────┘
   └───────────────────┘                                                  │ clicks
                                                                          ▼
   RGB-D frame ──▶ WildDet3D (click prompt, metric depth in) ─┬─▶ 2D/3D box → object geometry ┐
                                                              └─▶ 16x16 masked patch tokens   │   ┌────────────────┐
   robot URDF + joint state ──▶ projected silhouette ──▶ (kept in the patch mask)             ├──▶│ GAM policy     │──▶ action
   robot state ────────────────────────────────────────────────────────────────────────────────┘   │ (flow matching)│    chunk
                                                                                                    └────────────────┘
```

This README is the setup guide for running the stack on a real robot. Everything below the
adapter layer is hardware-agnostic; you write two small classes for your camera and robot.

---

## 0. What you need

| item | requirement |
| --- | --- |
| GPU | one 24 GB+ GPU for perception + policy (~10 GB), plus ~18 GB for the Molmo2-8B planner (same 48 GB card or a second GPU / machine) |
| RGB-D camera | colour + **depth aligned to colour**, metric depth (m), fixed mount, known intrinsics `K` |
| calibration | 4x4 pose of the camera in the **robot base frame** (hand-eye), OpenCV camera axes (+z forward, +y down) |
| robot | joint-position control at ~15 Hz; state/action layout **identical to the checkpoint's training data** (`policy.meta`, e.g. 14-d `[arm_L(6), gripper_L, arm_R(6), gripper_R]`) |
| robot model | the robot's URDF (+ meshes) and its joint names — projected into the camera to mask the robot in the image (see §4) |
| OS / Python | Linux, CUDA, Python ≥ 3.10, two conda envs (policy env, planner env) |

Checkpoints are trained for one robot embodiment, camera viewpoint and action layout: use a
checkpoint that matches your robot; do not expect a checkpoint trained on another platform to
transfer.

---

## 1. Install

```bash
git clone <this repo> gam && cd gam

# policy + perception environment
conda create -n gam python=3.11 -y && conda activate gam
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # your CUDA
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/co-tracker.git                # optional tracker
pip install -e .

# planner environment (Molmo2 needs transformers>=4.57; keep it separate)
conda create -n molmo2 python=3.11 -y && conda activate molmo2
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-planner.txt
```

`third_party/` contains the trimmed WildDet3D / SAM3 / LingBot-Depth code the patch extractor
needs; it is installed by `pip install -e .` (or found automatically when running from source).
On first use WildDet3D reads the LingBot-Depth model config from the HuggingFace hub
(`robbyant/lingbot-depth-postrain-dc-vitl14`); set `HF_HOME` if you use a shared cache.

## 2. Weights

```bash
bash scripts/download_weights.sh weights/     # CoTracker3 (optional)
```

| file (default path) | what | source |
| --- | --- | --- |
| `weights/gam_policy.pt` | GAM policy (release format) | provided with the checkpoint release |
| `weights/wilddet3d.pt` | WildDet3D RGB-D detector | provided with the checkpoint release |
| `weights/cotracker_scaled_online.pth` | CoTracker3 online (optional) | script above (Meta) |
| Molmo2-8B | planner VLM | pulled from `allenai/Molmo2-8B` on first planner start; `MOLMO2_MODEL=/local/path` to override |

If you converted your own training checkpoint: `python scripts/convert_checkpoint.py --src train.ckpt --dst weights/gam_policy.pt`.

## 3. Sanity check the software (no robot needed)

```bash
conda activate gam
python scripts/smoke_test.py --sample examples/sample_frame.npz \
    --policy weights/gam_policy.pt --wilddet weights/wilddet3d.pt
# expect: "... SMOKE_OK", an overlay examples/sample_frame_smoke.png and a per-step time
```

Then with the live planner (terminal 1 / terminal 2):

```bash
conda activate molmo2 && MOLMO2_MEM_DIR=runs/memory python -m gam.planner.server      # :18092, ~1 min to load
conda activate gam && python scripts/smoke_test.py --sample examples/sample_frame.npz \
    --policy weights/gam_policy.pt --wilddet weights/wilddet3d.pt \
    --planner http://127.0.0.1:18092 --observe_every 1 --steps 3
# expect: planner points at the object, memory STATE/FACTS lines, SMOKE_OK
```

The planner runs behind a plain HTTP API, so it can live on another machine: pass its URL in
`planner_url`. If your cluster sets `http_proxy`, the client already bypasses proxies for the
planner URL.

## 4. Write your adapters (`examples/adapters_template.py`)

Copy the template and implement:

```python
class MyCamera(gam.deploy.interfaces.Camera):
    def read(self) -> Frame   # rgb (H,W,3) uint8 RGB · depth_m (H,W) float32 metres, 0=invalid,
                              # aligned to rgb · K (3,3) · T_cam2base (4,4) camera pose in base frame

class MyRobot(gam.deploy.interfaces.Robot):
    def get_state(self) -> np.ndarray          # training layout, e.g. 14-d joints+grippers
    def apply_action(self, a: np.ndarray)      # ONE absolute joint target step, returns after ~1/hz
    def arm_mask(self, frame) -> np.ndarray    # (H,W) bool robot silhouette (URDF projection, below)
```

Robot silhouette from the URDF (train and test use the same projection):

```python
from gam.perception.arm_render import URDFArmMask          # pip install yourdfpy trimesh
from gam.perception.geometry import cam_to_world_to_E
mask = URDFArmMask("robot.urdf", joint_names=[...], T_base_root=np.eye(4),
                   link_filter=["arm_", "gripper_"])           # once; render only the arm links
def arm_mask(self, frame):
    H, W = frame.rgb.shape[:2]
    return mask(self.get_state(), frame.K, cam_to_world_to_E(frame.T_cam2base), H, W)
```

`joint_names` is the order of your state vector (entries that are not URDF joints, e.g.
gripper scalars, are ignored — name the gripper joints if you want the fingers to open/close in
the mask); `T_base_root` places the URDF root in the policy's base frame; `link_filter`
restricts rendering to the links whose names contain these substrings (rendering a whole
mobile-base URDF takes seconds, the two arms alone ~0.1 s per frame). Meshes are rasterised
triangle by triangle (plus a small morphological closing), so the silhouette follows the real
robot shape. If you already have your own FK, `MeshProjector` takes per-link meshes + per-link
poses instead. Install `fast-simplification` to let large visual meshes be decimated.

Notes that matter:

* **Frames.** Everything the policy sees in 3D (object centres, point clouds, box geometry) is
  expressed in the robot base frame through `T_cam2base`. A wrong calibration silently moves
  every object; verify it visually in §5 before moving the robot.
* **Robot silhouette is required.** The checkpoint was trained with the robot's projected
  silhouette kept in the patch mask; provide it via `arm_mask()` (URDF projection above, the same
  projection used to build the training data). Without it the commanded motion collapses to
  ~40 % on our reference frame.
* **State / action layout and units** must be exactly those of the training data (order of
  joints, gripper scale, radians). Print `policy.meta` (smoke test does) and compare.
* **Camera resolution / viewpoint.** Any resolution works: WildDet3D resamples internally
  (1008²), 2D boxes are rescaled into the checkpoint's training pixel space
  (`train_image_wh`), depth uses your `K`. Keep a similar field of view and viewpoint to the
  training camera and objects large enough to segment. Depth beyond a few metres is ignored.
* **Object geometry.** WildDet3D is prompted with the planner/tracker click and returns a 2D
  box (used for the patch mask) and an oriented 3D box. `geometry_source: box2d_depth` (default)
  back-projects the depth inside the 2D box into an axis-aligned object box — on our reference
  frame this reproduces the training-time action to within the training pipeline's own noise;
  `box3d` feeds WildDet3D's oriented 3D box instead (centre, size, orientation).
* **Colour order.** `image_channel_swap` in the checkpoint config records the colour order the
  backbone was trained with; the perception stack applies it automatically — always hand in
  RGB.

## 5. Check calibration, depth and masks on a captured frame (robot still)

```bash
python scripts/capture_frame.py --camera my_adapters.MyCamera --robot my_adapters.MyRobot \
    --instruction "put the red cup on the tray" --out captures/frame_001.npz
python scripts/smoke_test.py --sample captures/frame_001.npz --policy weights/gam_policy.pt \
    --wilddet weights/wilddet3d.pt --planner http://127.0.0.1:18092
```

Open `captures/frame_001_smoke.png` and check:

1. **green cross** — the planner's click sits on the object the instruction refers to;
2. **blue box** — WildDet3D's 2D box is tight around that object;
3. **red box** — the 3D box computed from depth, projected back into the image, lands on the
   object (if it is shifted, `T_cam2base` or `K` is wrong; if it is missing, depth is invalid there);
4. **yellow** — the robot silhouette (URDF projection) covers the visible robot; if it is
   offset from the robot, `T_base_root` / joint order / calibration is wrong;
5. dimmed area — everything outside object boxes ∪ robot is not shown to the policy (this is
   intended);
6. printed `objects` — centre in metres in the base frame is where the object really is (tape
   measure test), extents plausible;
7. no `state dim ... does not match` warning; per-step time printed (WildDet3D forward + URDF
   projection + 10 denoising steps; the first step includes the planner, ~20 s). With
   `n_exec: 15` at 15 Hz a chunk covers 1 s of motion. Camera resolution barely matters for
   speed (WildDet3D resamples to 1008² internally); keep `link_filter` to the arm links so the
   URDF projection stays ~0.1 s.

Repeat with a few instructions / object placements.

## 6. Run

```bash
# terminal 1 — planner
conda activate molmo2 && MOLMO2_MEM_DIR=runs/memory python -m gam.planner.server
# terminal 2 — control loop
conda activate gam
python -m gam.deploy.run_real --config configs/deploy_example.yaml \
    --instruction "put the red cup on the tray" \
    --camera my_adapters.MyCamera --robot my_adapters.MyRobot --hz 15 --max_steps 600
```

The loop: read frame → `GAMAgent.act()` (planner on frame 0 and every `observe_every` steps,
tracking + WildDet3D boxes/patch tokens + URDF silhouette + policy) → execute `n_exec` (15) of the 16 predicted steps at
`--hz`, feeding the measured state back → repeat. Chunk seams are blended (grippers excluded).

Start with the robot in the workspace pose the training data starts from, objects inside the
camera view, and a hand on the e-stop: the policy outputs absolute joint targets and this loop
adds no safety layer beyond your `apply_action` (put joint/velocity limits there).

`configs/deploy_example.yaml` (all `AgentConfig` fields):

| key | default | meaning |
| --- | --- | --- |
| `geometry_source` | `box2d_depth` | object geometry from WildDet3D's 2D box + depth (axis-aligned; matched the training geometry best in our tests) or `box3d` (WildDet3D oriented 3D box) |
| `n_points` | 512 | points per object slot |
| `observe_every` | 30 | steps between planner memory updates (0 = frame 0 only) |
| `n_exec` | 15 | executed steps per 16-step chunk |
| `smooth_seam` | true | blend first step of a chunk with the previous chunk's prediction |
| `tracker_ckpt` | — | CoTracker3 checkpoint; without it planner points are held and re-pointed by the planner when lost |
| `seed` | — | fix the sampling noise for reproducibility |

## 7. Planner: plan, progress, memory

`POST /session` (frame 0): Molmo2 writes an ordered **plan** of sub-steps ("grasp the left cup,
put it over the green block", ...), points at the first step's pick / place targets and writes
a one-line state note. Every planner point is an **object entity** that your tracker follows.

`POST /observe` (every `observe_every` control steps): receives the tracked pixels plus the
number of **gripper release events** since the last call. Each release completes one sub-step
(`MOLMO2_STEP_CHECK=gripper`, default — a release ends every pick-and-place step; a model-based
YES/NO check is available as `vlm` / `both` but was unreliable in our tests). On a step change
the planner records the completed step as a fact, asks *which visible object to grasp and where
to put it* for the new step (reasoning over the memory, e.g. "the red block is under the middle
cup"), points at them and returns `new_targets`; the agent tracks those and routes them to the
policy's pick / place slots. Lost objects are re-pointed. `GET /memory?session_id=` dumps plan,
progress, facts and object histories; `MOLMO2_MEM_DIR` persists one JSON per episode.

Measured on one A40 (bf16, HF transformers): `/session` 8–13 s (plan + 2 points + note),
`/observe` ≈2.3 s with 2 history frames (1.2 s single frame), one pointing query ≈1.2 s, a step
change +≈3 s; the server warms up once at start (~45 s). Env vars: `MOLMO2_MODEL`,
`MOLMO2_PORT`, `MOLMO2_DTYPE` (bf16), `MOLMO2_STEP_CHECK`, `MOLMO2_VERBOSE` (long state notes),
`MOLMO2_MEM_DIR`.

## 8. Troubleshooting

| symptom | check |
| --- | --- |
| planner OOM | it loads in bf16 (~17 GB); use a 24 GB+ GPU or another machine (`planner_url`) |
| `state dim … does not match` | your `get_state()` layout ≠ checkpoint; see `policy.meta` |
| red 3D box off the object | hand-eye calibration / intrinsics / depth alignment (§5) |
| tiny motions, robot barely moves | robot silhouette missing or misplaced (`arm_mask` None / URDF joint order / `T_base_root`) |
| planner clicks the wrong thing | phrase the instruction like the training data ("put the X on the Y"); check the JPEG the planner sees is RGB (not BGR) |
| slow steps | first step includes the planner (~20 s); later steps ≈1.2 s on our test GPU — WildDet3D forward dominates; use `link_filter` so the URDF projection renders only the arms |
| CoTracker not found | optional; install from GitHub or leave `tracker_ckpt` empty |

## Tests

```bash
python -m pytest -q tests                      # unit tests, no weights needed (~20 s)
GAM_TEST_POLICY=weights/gam_policy.pt GAM_TEST_WILDDET=weights/wilddet3d.pt python -m pytest -q tests
#                                              # + offline end-to-end agent test on the bundled frame
```

`tests/test_geometry.py` (projection / depth / OBB crops / observation builder),
`tests/test_model.py` (encoder / transformer shapes, normalizer, checkpoint round trip),
`tests/test_planner_and_render.py` (Molmo2 point parser, mock planner, mesh silhouette),
`tests/test_agent_offline.py` (full agent on `examples/sample_frame.npz`, needs weights).

## Repository layout

| path | what |
| --- | --- |
| `gam/model/` | policy: `encoder.py` (patch / object / state tokens + fusion), `transformer.py` (multi-stream MMDiT velocity net), `policy.py` (Euler flow sampling, checkpoint I/O) |
| `gam/perception/` | `patch_tokens.py` (WildDet3D → masked patch tokens + 2D/3D boxes), `arm_render.py` (URDF / mesh → robot silhouette), `geometry.py` (depth ↔ base frame, OBB crops, observation builder), `tracker.py` |
| `gam/planner/` | `server.py` Molmo2 planner with memory, `client.py` (+ `MockPlanner` for offline tests) |
| `gam/deploy/` | `agent.py` (`GAMAgent`, one control step), `interfaces.py` (Camera / Robot), `run_real.py` |
| `scripts/` | `smoke_test.py`, `capture_frame.py`, `convert_checkpoint.py`, `download_weights.sh` |
| `examples/` | `adapters_template.py`, `sample_frame.npz` |
| `third_party/` | vendored `wilddet3d`, `sam3`, `mdm` |

Checkpoint format: `gam_policy.pt = {"config", "state_dict", "normalizer"}`;
`GAMPolicy.from_checkpoint()` rebuilds the model from `config` (action/state dims, token layout,
`image_channel_swap`, `train_image_wh`, `meta` with the state/action layout description).
