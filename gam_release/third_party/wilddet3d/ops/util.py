"""Op utility functions."""

from __future__ import annotations

from functools import partial

import torch.nn.functional as F
from torch import Tensor


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments."""
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def flat_interpolate(
    flat_tensor: Tensor,
    old: tuple[int, int],
    new: tuple[int, int],
    antialias: bool = True,
    mode: str = "bilinear",
) -> Tensor:
    if old[0] == new[0] and old[1] == new[1]:
        return flat_tensor
    tensor = flat_tensor.view(
        flat_tensor.shape[0], old[0], old[1], -1
    ).permute(
        0, 3, 1, 2
    )
    tensor_interp = F.interpolate(
        tensor,
        size=(new[0], new[1]),
        mode=mode,
        align_corners=False,
        antialias=antialias,
    )
    flat_tensor_interp = tensor_interp.view(
        flat_tensor.shape[0], -1, new[0] * new[1]
    ).permute(
        0, 2, 1
    )
    return flat_tensor_interp.contiguous()
