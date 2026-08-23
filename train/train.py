"""Flow-matching training of the GAM policy on a precomputed cache.

    python train/train.py --config configs/my_task.yaml

Loss: v-prediction MSE with t ~ 0.999*Beta(1, 1.5), x_t = (1-t) x0 + t x1, dt = 0.
EMA (power 0.75) weights are what gets saved. Checkpoints are written in the release
format, directly loadable by GAMPolicy.from_checkpoint / eval scripts.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.config import load_config  # noqa: E402
from train.dataset import CacheDataset, fit_limits_normalizer  # noqa: E402
from gam.model.policy import GAMPolicy  # noqa: E402


def build_policy(cfg, ds) -> tuple:
    t = cfg["train"]
    S = len(cfg["data"]["slots"])
    n_patch = int(cfg["perception"].get("patch_grid", 16)) ** 2
    assert ds.eps[0]["patches"].shape[1] == n_patch, \
        f"cache patches {ds.eps[0]['patches'].shape[1]} != patch_grid^2 {n_patch} — re-run precompute"
    tps = int(t.get("tokens_per_slot", 8))
    use_lang = bool(t.get("use_language", False))   # +1 whole-sentence T5 token in CondFuse
    config = dict(
        action_dim=int(ds.action_dim), horizon=int(t["horizon"]),
        n_action_steps=int(t["n_action_steps"]), num_inference_steps=10,
        n_emb=int(t.get("n_emb", 576)), n_layer=int(t.get("n_layer", 12)),
        n_head=int(t.get("n_head", 16)),
        stream_split=[n_patch, S * tps] + ([1] if use_lang else []) + [1],
        encoder=dict(n_patch=n_patch, patch_dim=1024, n_slots=S, tokens_per_slot=tps,
                     fourier_L=8, pcd_dim=int(t.get("pcd_dim", 512)),
                     state_dim=int(ds.action_dim) * int(t["state_hist"]),
                     fuse_heads=16, fuse_layers=1,
                     state_dropout=float(t.get("state_dropout", 0.0)),
                     use_language=use_lang),
        image_channel_swap=bool(cfg["perception"].get("image_channel_swap", False)),
        image_wh=list(ds.image_wh), train_image_wh=list(ds.image_wh),
        attn_dropout=float(t.get("attn_dropout", 0.0)),
        meta=dict(state_hist=int(t["state_hist"]), n_points=int(cfg["perception"].get("n_points", 512)),
                  use_language=use_lang,
                  action=f"absolute joint targets, {ds.action_dim}-d", slots=cfg["data"]["slots"]),
    )
    policy = GAMPolicy(**{k: v for k, v in config.items() if k != "meta"})
    policy.meta = config["meta"]
    return policy, config


def optim_groups(policy, wd):
    """Reference-exact grouping: trunk Linear/MultiheadAttention weights decay; biases,
    LayerNorm/Embedding/RmsNorm weights and bare Parameters (pos embeddings, null tokens)
    do not; the observation encoder is one group WITH decay (as in the original recipe)."""
    import torch.nn as nn
    from timm.models.vision_transformer import RmsNorm
    decay, no_decay = [], []
    seen = set()
    for m in policy.model.modules():
        for pn, p in m.named_parameters(recurse=False):
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            if pn.endswith("bias") or pn.startswith("bias"):
                no_decay.append(p)
            elif isinstance(m, (nn.Linear, nn.MultiheadAttention)):
                decay.append(p)
            elif isinstance(m, (nn.LayerNorm, nn.Embedding, RmsNorm)):
                no_decay.append(p)
            else:                                    # bare Parameters: pos_emb, cond pos embed
                no_decay.append(p)
    enc = [p for p in policy.encoder.parameters() if p.requires_grad]
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
            {"params": enc, "weight_decay": wd}]


def ema_decay(step, power=0.75, max_value=0.9999):
    if step <= 0:
        return 0.0
    return min(max_value, 1 - (1 + step) ** -power)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=0, help="override train.max_steps")
    args = ap.parse_args()
    cfg = load_config(args.config)
    t = cfg["train"]
    torch.manual_seed(int(t.get("seed", 42)))
    np.random.seed(int(t.get("seed", 42)))
    out_dir = t["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    all_names = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in __import__("glob").glob(os.path.join(t["cache_dir"], "*.npz")))
    n_val = int(cfg["data"].get("val_episodes", 0))
    train_names = all_names[:-n_val] if n_val else all_names
    ds = CacheDataset(t["cache_dir"], train_names, horizon=int(t["horizon"]),
                      state_hist=int(t["state_hist"]))
    print(f"[train] {len(ds.eps)} episodes / {len(ds)} windows "
          f"(val held out: {all_names[len(train_names):]})", flush=True)
    dl = DataLoader(ds, batch_size=int(t["batch_size"]), shuffle=True,
                    num_workers=int(t.get("num_workers", 8)), pin_memory=False, drop_last=True)

    policy, config = build_policy(cfg, ds)
    scale, offset = fit_limits_normalizer(ds.all_actions())
    policy.normalizer.add("action", scale, offset)
    policy = policy.cuda().train()
    print(f"[train] params {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M", flush=True)

    max_steps = args.max_steps or int(t["max_steps"])
    warmup = int(t.get("warmup_steps", 1000))
    opt = torch.optim.AdamW(optim_groups(policy, float(t.get("weight_decay", 1e-3))),
                            lr=float(t["lr"]), betas=(0.9, 0.95), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(warmup, 1) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(max_steps - warmup, 1))))
    ema = copy.deepcopy(policy).eval().requires_grad_(False)
    xyz_drop = float(t.get("det_xyz_dropout", 0.0))

    step, t0, run_loss = 0, time.time(), 0.0
    while step < max_steps:
        for batch in dl:
            obs = {k: v.cuda(non_blocking=True) for k, v in batch["obs"].items()}
            act = batch["action"].cuda(non_blocking=True)
            B = act.shape[0]
            if xyz_drop > 0:                                # object 3D-centre dropout per sample
                keep = (torch.rand(B, 1, 1, device=act.device) >= xyz_drop).float()
                obs["xyz"] = obs["xyz"] * keep
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cond = policy.encoder(policy.normalizer.normalize(obs))
                x1 = policy.normalizer._affine("action", act, True)
                tt = 0.999 * torch.distributions.Beta(1.0, 1.5).sample((B,)).to(act.device)
                x0 = torch.randn_like(x1)
                x_t = (1 - tt.view(-1, 1, 1)) * x0 + tt.view(-1, 1, 1) * x1
                v = policy.model(x_t, tt, torch.zeros_like(tt), cond)
                loss = F.mse_loss(v.float(), (x1 - x0).float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            # EMA
            d = ema_decay(step)
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), policy.parameters()):
                    pe.mul_(d).add_(p.detach(), alpha=1 - d)
                for be, b in zip(ema.buffers(), policy.buffers()):
                    be.copy_(b)
            run_loss += loss.item()
            if step % 50 == 0:
                print(f"step {step}/{max_steps}  loss {run_loss / 50:.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {(time.time() - t0) / step:.2f}s/it",
                      flush=True)
                run_loss = 0.0
            if step % int(t.get("save_every", 1000)) == 0 or step == max_steps:
                ema.meta = policy.meta
                ema.save_checkpoint(os.path.join(out_dir, "latest.pt"), config)
                if step % (5 * int(t.get("save_every", 1000))) == 0:
                    ema.save_checkpoint(os.path.join(out_dir, f"step_{step}.pt"), config)
                print(f"[train] saved latest.pt @ step {step}", flush=True)
            if step >= max_steps:
                break
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
