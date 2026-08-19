"""Multi-stream MMDiT action transformer (flow-matching velocity network).

The action chunk (stream 0) and K condition streams (e.g. [patch, det, state]) each own
their adaLN / QKV / proj / MLP weights; attention runs once over the concatenated
sequence (SD3-style joint attention). Conditioning on the flow time `t` and the target
time `dt` enters through adaLN.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, RmsNorm


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MultiStreamBlock(nn.Module):
    """One joint-attention block over [action | cond_1 | ... | cond_K] with per-stream weights."""

    def __init__(self, hidden: int, num_heads: int, cond_split: Sequence[int],
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, qk_norm: bool = True,
                 attn_dropout: float = 0.0):
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.cond_split = tuple(int(v) for v in cond_split)
        S = 1 + len(self.cond_split)
        ln = lambda: nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)  # noqa: E731
        gelu = lambda: nn.GELU(approximate="tanh")  # noqa: E731
        self.norm1 = nn.ModuleList([ln() for _ in range(S)])
        self.norm2 = nn.ModuleList([ln() for _ in range(S)])
        self.qkv = nn.ModuleList([nn.Linear(hidden, hidden * 3, bias=qkv_bias) for _ in range(S)])
        self.proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(S)])
        self.mlp = nn.ModuleList([Mlp(hidden, int(hidden * mlp_ratio), act_layer=gelu, drop=0.0)
                                  for _ in range(S)])
        self.adaLN = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
                                    for _ in range(S)])
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_dropout = float(attn_dropout)          # active in training mode only

    def _heads(self, lin: nn.Linear, t: torch.Tensor):
        B, N, _ = t.shape
        q, k, v = lin(t).chunk(3, dim=-1)
        shp = (B, N, self.num_heads, self.head_dim)
        return (q.reshape(shp).transpose(1, 2), k.reshape(shp).transpose(1, 2),
                v.reshape(shp).transpose(1, 2))

    def forward(self, x: torch.Tensor, time_c: torch.Tensor, c: torch.Tensor):
        assert c.shape[1] == sum(self.cond_split), \
            f"cond len {c.shape[1]} != sum({self.cond_split})"
        parts = [x] + list(torch.split(c, self.cond_split, dim=1))
        mods = [ada(time_c).chunk(6, dim=-1) for ada in self.adaLN]
        qs, ks, vs = [], [], []
        for sid, p in enumerate(parts):
            n = modulate(self.norm1[sid](p), mods[sid][0], mods[sid][1])
            q, k, v = self._heads(self.qkv[sid], n)
            qs.append(q); ks.append(k); vs.append(v)
        q = self.q_norm(torch.cat(qs, dim=2))
        k = self.k_norm(torch.cat(ks, dim=2))
        v = torch.cat(vs, dim=2)
        o = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0
        ).transpose(1, 2)                                                # (B, N_total, H, hd)
        B = x.shape[0]
        outs = torch.split(o, [p.shape[1] for p in parts], dim=1)
        new = []
        for sid, (p, oi) in enumerate(zip(parts, outs)):
            oi = oi.reshape(B, p.shape[1], -1)
            p = p + mods[sid][2].unsqueeze(1) * self.proj[sid](oi)
            p = p + mods[sid][5].unsqueeze(1) * self.mlp[sid](
                modulate(self.norm2[sid](p), mods[sid][3], mods[sid][4]))
            new.append(p)
        return new[0], torch.cat(new[1:], dim=1)


class FinalLayer(nn.Module):
    def __init__(self, hidden: int, out_channels: int):
        super().__init__()
        self.norm_final = RmsNorm(hidden, eps=1e-6)
        self.ffn_final = Mlp(hidden, hidden, out_features=out_channels,
                             act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_final(self.norm_final(x))


class ActionTransformer(nn.Module):
    """Velocity network v(x_t, t, dt | cond) over an action chunk of length `horizon`.

    Args:
        action_dim: per-step action dimension.
        horizon: action chunk length T.
        cond_dim: dimension of the encoder's condition tokens.
        cond_len: number of condition tokens (= sum(stream_split)).
        stream_split: per-modality condition split, e.g. [256, 8, 1].
    """

    def __init__(self, action_dim: int, horizon: int, cond_dim: int, cond_len: int,
                 stream_split: Sequence[int], n_layer: int = 12, n_head: int = 16,
                 n_emb: int = 576, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 qk_norm: bool = True, t_embed_dim: int = 128, dt_embed_dim: int = 128,
                 attn_dropout: float = 0.0):
        super().__init__()
        assert sum(stream_split) == cond_len, (stream_split, cond_len)
        self.horizon = horizon
        self.hidden_dim = n_emb
        self.input_emb = nn.Linear(action_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.vis_cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.vis_cond_pos_embed = nn.Parameter(torch.zeros(1, cond_len, n_emb))
        self.flow_timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(t_embed_dim), nn.Linear(t_embed_dim, t_embed_dim * 4), nn.Mish(),
            nn.Linear(t_embed_dim * 4, n_emb))
        self.flow_target_t_encoder = nn.Sequential(
            SinusoidalPosEmb(dt_embed_dim), nn.Linear(dt_embed_dim, dt_embed_dim * 4), nn.Mish(),
            nn.Linear(dt_embed_dim * 4, n_emb))
        self.timestep_target_t_adaptor = nn.Linear(n_emb * 2, n_emb)
        self.blocks = nn.ModuleList([
            MultiStreamBlock(n_emb, n_head, stream_split, mlp_ratio, qkv_bias, qk_norm,
                             attn_dropout=attn_dropout)
            for _ in range(n_layer)])
        self.final_layer = FinalLayer(n_emb, action_dim)

    def forward(self, sample: torch.Tensor, t: torch.Tensor, dt: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """sample (B,T,Da) noisy actions; t, dt (B,) flow time / step; cond (B,L,cond_dim)."""
        x = self.input_emb(sample) + self.pos_emb
        B = sample.shape[0]
        t = t.expand(B) if t.dim() else t[None].expand(B)
        dt = dt.expand(B) if dt.dim() else dt[None].expand(B)
        time_c = torch.cat([self.flow_timestep_encoder(t), self.flow_target_t_encoder(dt)], -1)
        time_c = self.timestep_target_t_adaptor(time_c)
        c = self.vis_cond_obs_emb(cond) + self.vis_cond_pos_embed[:, :cond.shape[1]]
        for block in self.blocks:
            x, c = block(x, time_c, c)
        return self.final_layer(x)[:, -self.horizon:]
