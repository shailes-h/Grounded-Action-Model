"""Convert a research training checkpoint into the compact release format used by
gam.model.policy.GAMPolicy.from_checkpoint (config + state_dict + normalizer stats).

    python scripts/convert_checkpoint.py --src train.ckpt --dst gam_policy.pt
"""
import argparse
import re

import torch

RENAME = [
    (r"^obs_encoder\.rt_patch_proj\.", "encoder.patch_proj."),
    (r"^obs_encoder\._pcd_mlp\.", "encoder.pcd.mlp."),
    (r"^obs_encoder\._pcd_oct\.", "encoder.pcd.oct."),
    (r"^obs_encoder\._pcd_proj\.", "encoder.pcd.proj."),
    (r"^obs_encoder\.det_f5_proj\.", "encoder.det_proj."),
    (r"^obs_encoder\.det_sub_emb$", "encoder.det_sub_emb"),
    (r"^obs_encoder\.det_null$", "encoder.det_null"),
    (r"^obs_encoder\.det_role_emb\.", "encoder.det_role_emb."),
    (r"^obs_encoder\.det_cam_emb\.", "encoder.det_cam_emb."),
    (r"^obs_encoder\.state_proj\.", "encoder.state_proj."),
    (r"^obs_encoder\.cond_fuse\.", "encoder.fuse."),
    (r"^model\.", "model."),
]
STATE_KEY = "agent_pos_hist"      # training-time name of the proprioceptive history


def convert(src: str, dst: str):
    ck = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    cfg = ck["cfg"]
    sd = ck["state_dicts"].get("ema_model") or ck["state_dicts"]["model"]
    pol, enc = cfg.policy, cfg.policy.obs_encoder
    config = dict(
        action_dim=int(cfg.shape_meta.action.shape[0]), horizon=int(pol.horizon),
        n_action_steps=int(pol.n_action_steps), num_inference_steps=int(pol.num_inference_steps),
        n_emb=int(pol.n_emb), n_layer=int(pol.n_layer), n_head=int(pol.n_head),
        stream_split=[int(v) for v in pol.mmdit_stream_split],
        encoder=dict(n_patch=int(enc.rt_patch_tokens), patch_dim=int(enc.rt_patch_dim),
                     n_slots=int(enc.max_slots), tokens_per_slot=int(enc.det_token_multi),
                     fourier_L=int(enc.det_fourier_L), pcd_dim=int(enc.det_pcd_token),
                     state_dim=int(enc.observation_space.agent_pos[0]),
                     fuse_heads=int(enc.cond_fuse_n_heads), fuse_layers=int(enc.cond_fuse_n_layers)),
        image_channel_swap=True,          # this checkpoint's patch features were computed on BGR-ordered images
        image_wh=[float(v) for v in enc.click_native_wh],   # box normalisation constants
        train_image_wh=[1024.0, 768.0],   # pixel space the training boxes were expressed in
        meta=dict(n_points=int(enc.det_pcd_fps), state_hist=int(enc.state_hist),
                  action="absolute joint positions, 14-d: [arm_L(6), gripper_L, arm_R(6), gripper_R]",
                  state="last 2 joint states concatenated (oldest first), 28-d"),
    )
    out, dropped = {}, []
    for k, v in sd.items():
        nk = None
        for pat, rep in RENAME:
            if re.match(pat, k):
                nk = re.sub(pat, rep, k); break
        if nk is None:
            dropped.append(k); continue
        out[nk] = v.detach().clone()
    norm = {}
    for k in sd:
        m = re.match(r"^normalizer\.params_dict\.([^.]+)\.(scale|offset)$", k)
        if m:
            key = "action" if m.group(1) == "action" else ("state" if m.group(1) == STATE_KEY else None)
            if key is None:
                continue
            norm.setdefault(key, {})[m.group(2)] = sd[k].detach().clone()
    print(f"kept {len(out)} tensors, dropped {len(dropped)} (unused research heads / caches)")
    print("normalizer keys in ckpt:", sorted({re.match(r'^normalizer\.params_dict\.([^.]+)', k).group(1)
                                            for k in sd if k.startswith('normalizer.params_dict.')}))
    print("release normalizer:", {k: {kk: tuple(vv.shape) for kk, vv in v.items()} for k, v in norm.items()})
    torch.save({"config": config, "state_dict": out, "normalizer": norm}, dst)
    print("saved", dst)
    # sanity: instantiate + strict load
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from gam.model.policy import GAMPolicy
    p = GAMPolicy.from_checkpoint(dst, device="cpu")
    print("release policy params:", sum(v.numel() for v in p.parameters()) / 1e6, "M")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    convert(a.src, a.dst)
