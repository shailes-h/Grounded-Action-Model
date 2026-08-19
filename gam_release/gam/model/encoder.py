"""Observation encoder: masked patch tokens + object (det) tokens + state token -> fused
condition sequence for the action transformer.

Token layout per observation step (config `stream_split`, e.g. [256, 8, 1]):
    patch tokens   n_patch x d      linear projection of the 1024-d masked RGB-D patch tokens
    object tokens  S*M x d          per object slot: M read-out tokens from
                                    [box2d(4) | xyz(3) | Fourier(xyz) | point-cloud shape | box geometry(9)]
    state token    1 x d            proprioceptive history
followed by one self-attention "condition fusion" layer with modality embeddings.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn


class PointCloudEncoder(nn.Module):
    """Per-slot point-cloud shape token: per-point MLP -> masked max/mean pooling plus
    per-octant max pooling (octants w.r.t. the slot centroid) -> linear."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, dim), nn.LayerNorm(dim))
        self.oct = nn.Sequential(nn.Linear(8 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.proj = nn.Linear(3 * dim, dim)

    def forward(self, pts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """pts (B,K,N,3) world coordinates, mask (B,K,N) bool -> (B,K,dim); empty slot -> 0."""
        w = self.mlp[0].weight
        m = mask[..., None]
        cen = (pts * m.to(pts.dtype)).sum(2) / mask.sum(2).clamp(min=1)[..., None].to(pts.dtype)
        rel = pts - cen.unsqueeze(2)
        h = self.mlp(pts.to(w.dtype))
        neg = torch.finfo(h.dtype).min
        hmax = torch.where(m, h, torch.full_like(h, neg)).amax(dim=2)
        hmax = torch.where(mask.any(2)[..., None], hmax, torch.zeros_like(hmax))
        hmean = (h * m.to(h.dtype)).sum(2) / mask.sum(2).clamp(min=1)[..., None].to(h.dtype)
        oid = ((rel[..., 0] > 0).long() + (rel[..., 1] > 0).long() * 2 + (rel[..., 2] > 0).long() * 4)
        octs = []
        for o in range(8):
            mo = (mask & (oid == o))[..., None]
            ho = torch.where(mo, h, torch.full_like(h, neg)).amax(dim=2)
            octs.append(torch.where(mo.any(2), ho, torch.zeros_like(ho)))
        h_oct = self.oct(torch.cat(octs, dim=-1).to(w.dtype))
        tok = self.proj(torch.cat([hmax, hmean, h_oct], dim=-1))
        return torch.where(mask.any(2)[..., None], tok, torch.zeros_like(tok))


class _SABlock(nn.Module):
    def __init__(self, d: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, bias=True, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(int(d * mlp_ratio), d), nn.Dropout(dropout))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ConditionFusion(nn.Module):
    """Self-attention over all condition tokens with a learned modality embedding."""

    def __init__(self, d: int, heads: int, tokens_per_modality: Sequence[int],
                 n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        ids = []
        for mid, n in enumerate(tokens_per_modality):
            ids.extend([mid] * n)
        self.register_buffer("modality_ids", torch.tensor(ids, dtype=torch.long))
        self.modality_emb = nn.Embedding(len(tokens_per_modality), d)
        self.blocks = nn.ModuleList([_SABlock(d, heads, dropout=dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.modality_emb(self.modality_ids).unsqueeze(0)
        for b in self.blocks:
            x = b(x)
        return self.final_norm(x)


class ObsEncoder(nn.Module):
    def __init__(self, d: int = 576, n_patch: int = 256, patch_dim: int = 1024, n_slots: int = 1,
                 tokens_per_slot: int = 8, fourier_L: int = 8, pcd_dim: int = 512,
                 state_dim: int = 28, image_wh: Sequence[float] = (320.0, 240.0),
                 fuse_heads: int = 16, fuse_layers: int = 1, state_dropout: float = 0.0):
        super().__init__()
        self.d, self.n_patch, self.n_slots, self.M = d, n_patch, n_slots, tokens_per_slot
        self.image_wh = (float(image_wh[0]), float(image_wh[1]))
        self.register_buffer("fourier_freqs", (2.0 ** torch.arange(fourier_L).float()) * torch.pi,
                             persistent=False)
        self.patch_proj = nn.Linear(patch_dim, d, bias=False)
        self.pcd = PointCloudEncoder(pcd_dim)
        det_in = 7 + 3 * 2 * fourier_L + pcd_dim + 9
        self.det_proj = nn.Linear(det_in, d * tokens_per_slot)
        self.det_sub_emb = nn.Parameter(torch.zeros(tokens_per_slot, d))
        self.det_null = nn.Parameter(torch.zeros(1, 1, d))
        self.det_role_emb = nn.Embedding(n_slots, d)
        self.det_cam_emb = nn.Embedding(2, d)
        self.state_proj = nn.Sequential(nn.Linear(state_dim, d), nn.LayerNorm(d))
        self.state_dropout = float(state_dropout)   # training-only: whole state token zeroed
        self.fuse = ConditionFusion(d, fuse_heads, [n_patch, n_slots * tokens_per_slot, 1], fuse_layers)
        self.cond_len = n_patch + n_slots * tokens_per_slot + 1

    def _fourier(self, x):
        a = x.unsqueeze(-1) * self.fourier_freqs
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1).flatten(-2)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """obs (all (B,·)): patch_tokens (B,n_patch,patch_dim); box2d (B,S,4) pixel xyxy
        (zeros = empty slot); xyz (B,S,3); boxgeo (B,S,9); pcd_pts (B,S,N,3); pcd_mask
        (B,S,N); state (B,state_dim). Returns condition tokens (B, cond_len, d)."""
        dev = self.det_proj.weight.device
        B = obs["state"].shape[0]
        S = self.n_slots
        cw, ch = self.image_wh
        pd = (cw - ch) / 2.0                                    # letterbox to a square
        gb = obs["box2d"].reshape(B, S, 4).float().to(dev)
        b01 = torch.stack([gb[..., 0] / cw, (gb[..., 1] + pd) / cw,
                           gb[..., 2] / cw, (gb[..., 3] + pd) / cw], -1)
        valid = gb.abs().sum(-1) > 0                            # (B,S)
        xyz = obs["xyz"].reshape(B, S, 3).float().to(dev)
        boxgeo = obs["boxgeo"].reshape(B, S, 9).float().to(dev)
        pts = obs["pcd_pts"].reshape(B, S, -1, 3).float().to(dev)
        pmask = obs["pcd_mask"].reshape(B, S, -1).to(dev).bool()
        shape_tok = self.pcd(pts, pmask)                        # (B,S,pcd_dim)
        feats = torch.cat([b01, xyz, self._fourier(xyz), shape_tok.to(xyz.dtype), boxgeo], -1)
        tok = self.det_proj(feats.to(self.det_proj.weight.dtype)).to(torch.float32)
        tok = tok.reshape(B, S, self.M, self.d) + self.det_sub_emb[None, None].to(tok.dtype)
        roles = torch.arange(S, device=dev)
        tok = tok + self.det_role_emb(roles)[None, :, None].to(tok.dtype) \
                  + self.det_cam_emb(torch.tensor(0, device=dev)).to(tok.dtype)
        null = self.det_null.to(tok.dtype)[:, :, None, :].expand(B, S, self.M, self.d)
        det_seq = torch.where(valid[:, :, None, None], tok, null).reshape(B, S * self.M, self.d)
        state_tok = self.state_proj(obs["state"].reshape(B, -1).float().to(dev)).unsqueeze(1)
        if self.training and self.state_dropout > 0:
            keep = (torch.rand(B, 1, 1, device=dev) >= self.state_dropout).to(state_tok.dtype)
            state_tok = state_tok * keep
        ptok = self.patch_proj(obs["patch_tokens"].reshape(B, self.n_patch, -1).float().to(dev))
        scene = torch.cat([ptok.to(state_tok.dtype), det_seq.to(state_tok.dtype), state_tok], 1)
        return self.fuse(scene)
