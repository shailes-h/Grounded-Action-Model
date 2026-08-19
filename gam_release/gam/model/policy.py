"""Flow-matching action policy: ObsEncoder -> ActionTransformer, Euler ODE sampling."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .encoder import ObsEncoder
from .normalizer import Normalizer
from .transformer import ActionTransformer


class GAMPolicy(nn.Module):
    """Args mirror the released checkpoint config (`config` dict inside the .pt file)."""

    def __init__(self, action_dim: int = 14, horizon: int = 16, n_action_steps: int = 16,
                 num_inference_steps: int = 10, n_emb: int = 576, n_layer: int = 12,
                 n_head: int = 16, stream_split=(256, 8, 1), encoder: dict | None = None,
                 image_channel_swap: bool = False, image_wh=(320.0, 240.0),
                 train_image_wh=(1024.0, 768.0), attn_dropout: float = 0.0):
        super().__init__()
        enc = dict(encoder or {})
        enc.setdefault("d", n_emb)
        enc.setdefault("image_wh", image_wh)
        self.encoder = ObsEncoder(**enc)
        assert self.encoder.cond_len == sum(stream_split), (self.encoder.cond_len, stream_split)
        self.model = ActionTransformer(action_dim, horizon, n_emb, self.encoder.cond_len,
                                       stream_split, n_layer=n_layer, n_head=n_head, n_emb=n_emb,
                                       attn_dropout=attn_dropout)
        self.normalizer = Normalizer()
        self.action_dim, self.horizon = action_dim, horizon
        self.n_action_steps, self.num_inference_steps = n_action_steps, num_inference_steps
        self.image_channel_swap = bool(image_channel_swap)   # perception hint (see PatchTokenExtractor)
        self.image_wh = tuple(image_wh)
        self.train_image_wh = tuple(train_image_wh)     # 2D boxes must be scaled into this pixel space

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, generator=None) -> torch.Tensor:
        B = cond.shape[0]
        x = torch.randn(B, self.horizon, self.action_dim, device=cond.device, generator=generator)
        N = self.num_inference_steps
        dt = 1.0 / N
        ts = torch.arange(0, N, device=cond.device, dtype=torch.float32) / N
        dt_t = torch.tensor([dt], dtype=torch.float32, device=cond.device)
        for i in range(N):
            ti = torch.ones((B,), device=cond.device) * ts[i]
            v = self.model(x, ti, dt_t, cond)
            x = x.detach().clone() + v * dt
        return x

    @torch.no_grad()
    def predict_action(self, obs: Dict[str, torch.Tensor], generator=None) -> torch.Tensor:
        """obs: see ObsEncoder.forward (unnormalized). Returns actions (B, n_action_steps, Da)."""
        nobs = self.normalizer.normalize(obs)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cond = self.encoder(nobs)
            x = self.sample(cond, generator)
        act = self.normalizer.unnormalize("action", x[..., :self.action_dim].float())
        return act[:, :self.n_action_steps]

    # ---- serialization -------------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda") -> "GAMPolicy":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(ck["config"])
        meta = cfg.pop("meta", None)
        norm = ck.get("normalizer", {})
        policy = cls(**cfg)
        policy.meta = meta or {}
        for k, v in norm.items():
            policy.normalizer.add(k, torch.as_tensor(v["scale"]), torch.as_tensor(v["offset"]))
        missing, unexpected = policy.load_state_dict(ck["state_dict"], strict=False)
        missing = [m for m in missing if not m.startswith("normalizer.")]
        assert not missing and not unexpected, (missing, unexpected)
        return policy.to(device).eval()

    def save_checkpoint(self, path: str, config: dict):
        norm = {k: {"scale": getattr(self.normalizer, f"{k}_scale").cpu(),
                    "offset": getattr(self.normalizer, f"{k}_offset").cpu()} for k in self.normalizer.keys}
        sd = {k: v for k, v in self.state_dict().items() if not k.startswith("normalizer.")}
        torch.save({"config": config, "state_dict": sd, "normalizer": norm}, path)
