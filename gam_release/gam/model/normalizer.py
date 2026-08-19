"""Per-key affine normalizer (x_norm = x * scale + offset), stats stored as buffers."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    def __init__(self, stats: Dict[str, Dict[str, torch.Tensor]] | None = None):
        super().__init__()
        self.keys = []
        if stats:
            for k, v in stats.items():
                self.add(k, v["scale"], v["offset"])

    def add(self, key: str, scale: torch.Tensor, offset: torch.Tensor):
        dev = next(iter(self.buffers()), torch.zeros(0)).device      # follow the module's device
        self.register_buffer(f"{key}_scale", scale.detach().clone().float().to(dev))
        self.register_buffer(f"{key}_offset", offset.detach().clone().float().to(dev))
        if key not in self.keys:
            self.keys.append(key)

    def _affine(self, key: str, x: torch.Tensor, forward: bool) -> torch.Tensor:
        s = getattr(self, f"{key}_scale").to(x.device); o = getattr(self, f"{key}_offset").to(x.device)
        shp = x.shape
        x = x.to(s.dtype).reshape(-1, s.shape[0])
        x = x * s + o if forward else (x - o) / s
        return x.reshape(shp)

    def normalize(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: (self._affine(k, v, True) if k in self.keys else v) for k, v in d.items()}

    def unnormalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return self._affine(key, x, False)
