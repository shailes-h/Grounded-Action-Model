"""Offline (open-loop) evaluation on held-out cached episodes: predict a 16-step chunk at
every K-th frame and compare with the recorded actions. No simulator or robot needed.

    python eval/eval_offline.py --config configs/my_task.yaml --ckpt runs/yam_task/latest.pt

Reports per-arm / per-gripper MAE and writes joint-trajectory comparison plots. Low MAE on
held-out episodes means the policy imitates the data; closed-loop success still requires
eval/deploy_policy.py in your environment.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.config import load_config  # noqa: E402
from train.dataset import CacheDataset  # noqa: E402
from gam.model.policy import GAMPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--every", type=int, default=15, help="predict a chunk every N frames")
    args = ap.parse_args()
    cfg = load_config(args.config)
    t = cfg["train"]
    import glob
    all_names = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(t["cache_dir"], "*.npz")))
    n_val = int(cfg["data"].get("val_episodes", 0)) or 1
    val_names = all_names[-n_val:]
    ds = CacheDataset(t["cache_dir"], val_names, horizon=int(t["horizon"]),
                      state_hist=int(t["state_hist"]))
    policy = GAMPolicy.from_checkpoint(args.ckpt)
    print(f"[eval] ckpt meta: {policy.meta} | val episodes: {val_names}", flush=True)
    gen = torch.Generator(device="cuda").manual_seed(0)
    errs, curves = [], {}
    for e, name in enumerate(ds.names):
        ep = ds.eps[e]
        T = len(ep["state"])
        pred_traj = np.full_like(ep["action"], np.nan)
        for t0 in range(0, T - 1, args.every):
            obs = {k: v[None].cuda() for k, v in ds.obs_at(e, t0).items()}
            a = policy.predict_action(obs, generator=gen)[0].float().cpu().numpy()
            idx = np.clip(np.arange(t0, t0 + len(a)), 0, T - 1)
            gt = ep["action"][idx]
            errs.append(np.abs(a - gt))
            pred_traj[t0:min(t0 + len(a), T)] = a[:min(len(a), T - t0)]
        curves[name] = (ep["action"], pred_traj)
        print(f"[eval] {name}: {T} frames done", flush=True)
    E = np.concatenate(errs, 0)
    D = E.shape[-1]
    print(f"\n== open-loop MAE over {len(errs)} chunks (val) ==")
    print(f"all dims: {E.mean():.4f}")
    for i in range(D):
        print(f"  dim {i:2d}: {E[..., i].mean():.4f}")
    out = os.path.join(t["out_dir"], "eval_offline")
    os.makedirs(out, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name, (gt, pr) in curves.items():
            fig, axes = plt.subplots(D, 1, figsize=(10, 1.4 * D), sharex=True)
            for i in range(D):
                axes[i].plot(gt[:, i], lw=1, label="recorded")
                axes[i].plot(pr[:, i], lw=1, ls="--", label="predicted")
                axes[i].set_ylabel(f"d{i}", fontsize=7)
            axes[0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"{name}.png"), dpi=110)
            plt.close(fig)
        print(f"[eval] trajectory plots -> {out}/")
    except Exception as ex:  # noqa: BLE001
        print(f"[eval] plots skipped ({ex})")
    print("EVAL_OFFLINE_DONE")


if __name__ == "__main__":
    main()
