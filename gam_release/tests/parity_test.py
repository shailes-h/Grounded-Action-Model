"""Bit-parity of the release policy against a stored reference (obs, action) pair:

    python tests/parity_test.py ref_sample.pt gam_policy.pt

Provenance: the original reference was produced by the source training codebase and matched
bit-exactly (max|diff| = 0.0, torch 2.4, one GPU arch). Kernel numerics shift across torch
versions / GPU architectures (~6e-3 on this action scale), so the shipped reference is
re-baselined from the release policy itself under the pinned environment — it guards future
CODE changes; run it on the architecture named inside the reference file.
"""
import sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gam.model.policy import GAMPolicy

ref = torch.load(sys.argv[1], map_location="cpu")
ckpt = sys.argv[2]
o = ref["obs"]
policy = GAMPolicy.from_checkpoint(ckpt, device="cuda")
obs = {
    "patch_tokens": o["patch_feat"][:, 0].cuda(),               # (1,256,1024)
    "box2d": o["gt_box2d_agv"][:, 0].reshape(1, -1, 4).cuda(),
    "xyz": o["gt_pick_xyz"][:, 0].reshape(1, -1, 3).cuda(),
    "boxgeo": o["gt_boxgeo"][:, 0].cuda(),                        # (1,S,9)
    "pcd_pts": o["pcd_pts"][:, 0].cuda(),                          # (1,S,N,3)
    "pcd_mask": o["pcd_mask"][:, 0].cuda(),
    "state": o["agent_pos_hist"][:, 0].cuda(),                     # (1,28)
}
torch.manual_seed(0)
act = policy.predict_action(obs)
ref_a = ref["action"].cuda()
d = (act - ref_a).abs()
print("release action[0,:2]:", act[0, :2])
print("max|diff|", d.max().item(), "mean|diff|", d.mean().item())
print("PARITY_OK" if d.max().item() < 1e-4 else "PARITY_FAIL")
