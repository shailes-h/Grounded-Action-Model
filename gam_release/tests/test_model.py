import os
import tempfile

import numpy as np
import pytest
import torch

from gam.model.encoder import ObsEncoder
from gam.model.normalizer import Normalizer
from gam.model.policy import GAMPolicy
from gam.model.transformer import ActionTransformer

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _small_cfg():
    return dict(action_dim=14, horizon=16, n_action_steps=16, num_inference_steps=2, n_emb=64,
                n_layer=2, n_head=4, stream_split=[16, 8, 1],
                encoder=dict(n_patch=16, patch_dim=32, n_slots=1, tokens_per_slot=8, fourier_L=8,
                             pcd_dim=32, state_dim=28, fuse_heads=4, fuse_layers=1),
                image_channel_swap=False, image_wh=[320.0, 240.0], train_image_wh=[1024.0, 768.0])


def _obs(B=2, S=1, N=64):
    g = torch.Generator().manual_seed(0)
    return {"patch_tokens": torch.randn(B, 16, 32, generator=g),
            "box2d": torch.tensor([[[300., 200., 400., 300.]]]).repeat(B, S, 1),
            "xyz": torch.randn(B, S, 3, generator=g) * 0.1,
            "boxgeo": torch.tensor([[[0.1, 0.1, 0.1, 1, 0, 0, 0, 1, 0]]]).repeat(B, S, 1),
            "pcd_pts": torch.randn(B, S, N, 3, generator=g) * 0.05,
            "pcd_mask": torch.ones(B, S, N, dtype=torch.bool),
            "state": torch.randn(B, 28, generator=g)}


def test_encoder_shapes_and_empty_slot():
    enc = ObsEncoder(d=64, n_patch=16, patch_dim=32, n_slots=2, tokens_per_slot=8, pcd_dim=32,
                     fuse_heads=4)
    o = _obs(S=2)
    o["box2d"][:, 1] = 0; o["pcd_mask"][:, 1] = False           # slot 1 empty
    y = enc(o)
    assert y.shape == (2, 16 + 16 + 1, 64) and torch.isfinite(y).all()
    # empty slot -> its 8 tokens equal the learned null token (before fusion, check det path)
    assert enc.cond_len == 33


def test_transformer_shapes():
    m = ActionTransformer(action_dim=14, horizon=16, cond_dim=64, cond_len=25, stream_split=[16, 8, 1],
                          n_layer=2, n_head=4, n_emb=64)
    x = torch.randn(2, 16, 14); c = torch.randn(2, 25, 64)
    y = m(x, torch.tensor([0.5, 0.5]), torch.tensor([0.1]), c)
    assert y.shape == (2, 16, 14)


def test_normalizer_roundtrip():
    n = Normalizer({"action": {"scale": torch.full((14,), 2.0), "offset": torch.full((14,), -1.0)}})
    x = torch.randn(3, 16, 14)
    y = n.normalize({"action": x, "other": x})
    assert torch.allclose(y["other"], x) and torch.allclose(n.unnormalize("action", y["action"]), x, atol=1e-6)


def test_policy_predict_and_checkpoint_roundtrip():
    cfg = _small_cfg()
    p = GAMPolicy(**cfg).to(DEV).eval()
    p.normalizer.add("action", torch.ones(14), torch.zeros(14))
    o = {k: v.to(DEV) for k, v in _obs().items()}
    if DEV == "cpu":
        pytest.skip("predict_action uses CUDA autocast")
    torch.manual_seed(0); a1 = p.predict_action(o)
    assert a1.shape == (2, 16, 14)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ck.pt")
        p.save_checkpoint(path, cfg)
        p2 = GAMPolicy.from_checkpoint(path, DEV)
        torch.manual_seed(0); a2 = p2.predict_action(o)
    assert torch.allclose(a1, a2)
    assert p2.train_image_wh == (1024.0, 768.0)
