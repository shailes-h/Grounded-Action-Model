"""Offline end-to-end smoke test on a recorded RGB-D frame (no robot, no planner needed).

    python scripts/smoke_test.py --sample examples/sample_frame.npz \
        --policy weights/gam_policy.pt --wilddet weights/wilddet3d.pt \
        [--planner http://127.0.0.1:18092 | --click u v]

Runs the full agent (planner -> tracking -> boxes -> patch tokens -> policy) for two control
steps and writes an overlay image next to the sample.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gam.deploy.agent import AgentConfig, GAMAgent  # noqa: E402
from gam.deploy.interfaces import Frame  # noqa: E402
from gam.perception.geometry import box2d_from_obb, cam_to_world_to_E, project  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--wilddet", required=True)
    ap.add_argument("--planner", default=None, help="planner URL; omit to use --click")
    ap.add_argument("--click", nargs=2, type=float, default=None, help="pixel (u v) of the pick object")
    ap.add_argument("--geometry", default="box2d_depth", choices=["box2d_depth", "box3d"])
    ap.add_argument("--observe_every", type=int, default=0, help="planner memory update period (0=off)")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    s = np.load(args.sample, allow_pickle=True)
    rgb, depth = s["rgb"], s["depth_m"]
    frame = Frame(rgb=rgb, depth_m=depth, K=s["K"], T_cam2base=s["T_cam2base"])
    instruction = str(s["instruction"]) if "instruction" in s else "pick up the object"
    if args.planner:
        planner = args.planner
    else:
        c = args.click if args.click else s["ref_click"].tolist()
        planner = f"mock:{c[0]},{c[1]}"
    cfg = AgentConfig(policy_ckpt=args.policy, wilddet_ckpt=args.wilddet,
                      planner_url=planner, geometry_source=args.geometry, observe_every=args.observe_every, seed=0)
    agent = GAMAgent(cfg)
    print("[smoke] checkpoint meta:", agent.policy.meta,
          "| state_dim", agent.policy.encoder.state_proj[0].in_features // 2, "x2 | action_dim", agent.policy.action_dim)
    if s["q"].shape[0] * 2 != agent.policy.encoder.state_proj[0].in_features:
        print(f"[smoke] WARNING: robot state dim {s['q'].shape[0]} does not match the checkpoint")
    agent.reset(instruction)
    q = s["q"]
    if "q_prev" in s:
        agent.push_state(s["q_prev"]); agent.push_state(q)
    arm_mask = s["arm_mask"].astype(bool) if "arm_mask" in s else None
    if arm_mask is None:
        print("[smoke] WARNING: sample has no robot mask (arm_mask) — the policy expects one")
    import time
    for step in range(args.steps):
        t0 = time.time()
        chunk = agent.act(frame, q, arm_mask=arm_mask)
        print(f"[smoke] step {step}: chunk {chunk.shape}, first action {np.round(chunk[0], 3)}"
              f"  ({time.time() - t0:.2f}s{' incl. planner' if step == 0 else ''})")
        if "ref_action" in s and step == 0:
            ra = s["ref_action"][:len(chunk)]
            print(f"[smoke] |pred - recorded| mean over chunk: {np.abs(chunk - ra).mean():.4f} "
                  f"(recorded first action {np.round(ra[0], 3)})")
    last = agent.last
    if agent.session_id:
        print("[smoke] planner memory:", agent.memory_state)
    print("[smoke] clicks:", last["clicks"], "\n[smoke] objects:",
          {k: {kk: np.round(vv, 3).tolist() for kk, vv in v.items()} for k, v in last["objects"].items()})
    print("[smoke] patch cells kept:", int(last["patch_mask"].sum()), "/", last["patch_mask"].size)
    # overlay
    E = cam_to_world_to_E(frame.T_cam2base)
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    H, W = rgb.shape[:2]
    for s_, c in enumerate(last["clicks"]):
        if c is not None:
            cv2.drawMarker(vis, (int(c[0]), int(c[1])), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    for b in last["boxes"]:
        if b is not None and b[2] > b[0]:
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 160, 0), 2)
    for name, o in last["objects"].items():
        b2 = box2d_from_obb(o["p"], o["extents"], o["R"], E, frame.K)
        if b2.any():
            cv2.rectangle(vis, (int(b2[0]), int(b2[1])), (int(b2[2]), int(b2[3])), (0, 0, 255), 2)
            cv2.putText(vis, name, (int(b2[0]), max(int(b2[1]) - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
    if last.get("arm_mask") is not None:
        vis[last["arm_mask"]] = (0.5 * vis[last["arm_mask"]] + 0.5 * np.array([0, 200, 255])).astype(np.uint8)
    m = cv2.resize(last["patch_mask"].astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
    vis = np.where(m[..., None] > 0, vis, (vis * 0.35).astype(np.uint8))
    out = args.out or os.path.splitext(args.sample)[0] + "_smoke.png"
    cv2.imwrite(out, vis)
    print("[smoke] overlay ->", out)
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
