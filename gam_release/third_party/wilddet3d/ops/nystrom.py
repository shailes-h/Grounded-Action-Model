"""Nystrom Attention.

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""

import math
import warnings
from contextlib import nullcontext

import torch
from torch import Tensor, nn


class AvgPool(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x: Tensor):
        seq_len = x.shape[1]
        head_dim = x.shape[2]
        segments = seq_len // self.n
        assert (
            segments > 0
        ), "num_landmarks should be smaller than the sequence length"

        if seq_len % self.n == 0:
            return x.reshape(
                -1,
                self.n,
                segments,
                head_dim,
            ).mean(dim=-2)

        n_round = self.n - seq_len % self.n

        x_avg_round = (
            x[:, : n_round * segments, :]
            .reshape(-1, n_round, segments, head_dim)
            .mean(dim=-2)
        )
        x_avg_off = (
            x[:, n_round * segments :, :]
            .reshape(-1, self.n - n_round, segments + 1, head_dim)
            .mean(dim=-2)
        )
        return torch.cat((x_avg_round, x_avg_off), dim=-2)


def bmm(a: Tensor, b: Tensor) -> Tensor:
    return a @ b


def _apply_dropout(att, dropout):
    if dropout is None:
        return att
    att = dropout(att)
    return att


def _matmul_with_mask(
    a: Tensor,
    b: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    if mask is None:
        return a @ b

    att = a @ b
    if mask.dtype == torch.bool:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(att.shape[0], -1, -1)
        att[~mask] = float("-inf")
    else:
        if (
            mask.ndim == 3
            and mask.shape[0] != att.shape[0]
            and (att.shape[0] % mask.shape[0]) == 0
        ):
            repeat_factor = att.shape[0] // mask.shape[0]
            mask = mask.repeat([repeat_factor, 1, 1])
            warnings.warn(
                "Mismatched batch dimensions for mask, repeating mask."
            )
        att += mask
    return att


def _softmax(a: Tensor) -> Tensor:
    if a.is_sparse:
        return torch.sparse.softmax(a, dim=a.ndim - 1)
    return torch.softmax(a, dim=a.ndim - 1)


def scaled_query_key_softmax(
    q: Tensor,
    k: Tensor,
    att_mask: Tensor | None = None,
) -> Tensor:
    q = q / math.sqrt(k.size(-1))
    mask = att_mask
    att = _matmul_with_mask(q, k.transpose(-2, -1), mask)
    att = _softmax(att)
    return att


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    att_mask: Tensor | None = None,
    dropout: nn.Module | None = None,
) -> Tensor:
    autocast_disabled = att_mask is not None and att_mask.is_sparse

    with torch.cuda.amp.autocast(enabled=False) if autocast_disabled else nullcontext():
        if autocast_disabled:
            q, k, v = q.float(), k.float(), v.float()

        att = scaled_query_key_softmax(q, k, att_mask=att_mask)
        att = _apply_dropout(att, dropout)
        y = bmm(att, v)
    return y


def bool_mask_to_additive(
    mask: Tensor, dtype: torch.dtype | None = torch.float32
) -> Tensor:
    assert (
        mask.dtype == torch.bool
    ), "This util is meant to convert in between bool masks and additive ones"

    mask_ = torch.zeros_like(mask, dtype=dtype)
    mask_[~mask] = float("-inf")
    return mask_


def iterative_pinv(
    softmax_mat: Tensor, n_iter=6, pinverse_original_init=False
):
    """Computing the Moore-Penrose inverse via iterative method."""
    i = torch.eye(
        softmax_mat.size(-1),
        device=softmax_mat.device,
        dtype=softmax_mat.dtype,
    )
    k = softmax_mat

    if pinverse_original_init:
        v = 1 / torch.max(torch.sum(k, dim=-2)) * k.transpose(-1, -2)
    else:
        v = (
            1
            / torch.max(torch.sum(k, dim=-2), dim=-1).values[:, None, None]
            * k.transpose(-1, -2)
        )

    for _ in range(n_iter):
        kv = torch.matmul(k, v)
        v = torch.matmul(
            0.25 * v,
            13 * i - torch.matmul(kv, 15 * i - torch.matmul(kv, 7 * i - kv)),
        )
    return v


def reshape_key_padding_mask(
    key_padding_mask: Tensor, batched_dim: int
) -> Tensor:
    assert key_padding_mask.ndim == 2
    batch_size, src_len = key_padding_mask.size()
    num_heads = batched_dim // batch_size
    return _reshape_key_padding_mask(
        key_padding_mask, batch_size, src_len, num_heads
    )


def _reshape_key_padding_mask(
    key_padding_mask: Tensor,
    batch_size: int,
    src_len: int,
    num_heads: int,
) -> Tensor:
    assert key_padding_mask.shape == (batch_size, src_len)
    key_padding_mask = (
        key_padding_mask.view(batch_size, 1, 1, src_len)
        .expand(-1, num_heads, -1, -1)
        .reshape(batch_size * num_heads, 1, src_len)
    )
    return key_padding_mask


class NystromAttention(nn.Module):
    """Nystrom attention mechanism."""

    def __init__(
        self,
        dropout: float,
        num_heads: int,
        num_landmarks: int = 64,
        landmark_pooling: nn.Module | None = None,
        causal: bool = False,
        use_razavi_pinverse: bool = True,
        pinverse_original_init: bool = False,
        inv_iterations: int = 6,
        v_skip_connection: nn.Module | None = None,
        conv_kernel_size: int | int = None,
    ):
        """Creates an instance of the class."""
        super().__init__()
        self.requires_separate_masks = True
        self.num_landmarks = num_landmarks
        self.num_heads = num_heads
        self.use_razavi_pinverse = use_razavi_pinverse
        self.pinverse_original_init = pinverse_original_init
        self.inv_iterations = inv_iterations
        self.attn_drop = nn.Dropout(dropout)
        self.skip_connection = v_skip_connection
        self.causal = causal

        if self.skip_connection is None and conv_kernel_size is not None:
            self.skip_connection = nn.Conv2d(
                in_channels=self.num_heads,
                out_channels=self.num_heads,
                kernel_size=(conv_kernel_size, 1),
                padding=(conv_kernel_size // 2, 0),
                bias=False,
                groups=self.num_heads,
            )

        if landmark_pooling is not None:
            self.landmark_pooling = landmark_pooling
        else:
            self.landmark_pooling = AvgPool(n=self.num_landmarks)

        self.causal_mask_1: Tensor | None = None
        self.causal_mask_2: Tensor | None = None
        self.causal_mask_3: Tensor | None = None

        self.supports_attention_mask = False
        self.supports_key_padding_mask = True

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Tensor | None = None,
        *args,
        **kwargs,
    ):
        batched_dim = k.size(0)
        seq_len = k.size(-2)
        tt = {"dtype": q.dtype, "device": q.device}

        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                warnings.warn(
                    "Bool mask found, but an additive mask is expected. "
                    "Converting but this is slow"
                )
                key_padding_mask = bool_mask_to_additive(key_padding_mask)

            if key_padding_mask.ndim == 2:
                key_padding_mask = reshape_key_padding_mask(
                    key_padding_mask, batched_dim
                )

            zeros = torch.zeros_like(key_padding_mask)
            ones = torch.ones_like(key_padding_mask)
            is_masked = torch.isinf(-key_padding_mask)

            _mask = torch.where(is_masked, zeros, ones)
            _mask = _mask.transpose(2, 1)
            assert _mask.shape == (batched_dim, q.shape[1], 1)

            q = q * _mask
            k = k * _mask

            assert key_padding_mask.size() == (batched_dim, 1, seq_len), (
                f"key_padding_mask has invalid dimensions {key_padding_mask.size()}."
                f" Must have dimensions {batched_dim, 1, seq_len} or (batch_size, {seq_len})."
            )

        if self.num_landmarks >= seq_len:
            mask: Tensor | None = None

            if self.causal:
                mask = self._triu_mask(batched_dim, seq_len, seq_len, **tt)

            if key_padding_mask is not None:
                mask = (
                    key_padding_mask
                    if mask is None
                    else mask + key_padding_mask
                )

            x = scaled_dot_product_attention(q=q, k=k, v=v, att_mask=mask)

        else:
            q_landmarks = self.landmark_pooling(q)
            k_landmarks = self.landmark_pooling(k)

            if self.causal and (
                self.causal_mask_1 is None
                or (batched_dim, seq_len, self.num_landmarks)
                != self.causal_mask_1.size()
            ):
                self.causal_mask_1 = self._triu_mask(
                    batched_dim, seq_len, self.num_landmarks, **tt
                )
                self.causal_mask_2 = self._triu_mask(
                    batched_dim, self.num_landmarks, self.num_landmarks, **tt
                )
                self.causal_mask_3 = self._triu_mask(
                    batched_dim, self.num_landmarks, seq_len, **tt
                )

            mask_3: Tensor | None = self.causal_mask_3
            if key_padding_mask is not None:
                mask_3 = (
                    key_padding_mask
                    if mask_3 is None
                    else mask_3 + key_padding_mask
                )

            kernel_1 = scaled_query_key_softmax(
                q=q, k=k_landmarks, att_mask=None
            )
            kernel_2 = scaled_query_key_softmax(
                q=q_landmarks, k=k_landmarks, att_mask=None
            )
            kernel_3 = scaled_dot_product_attention(
                q=q_landmarks, k=k, v=v, att_mask=mask_3
            )

            kernel_2_inv = (
                iterative_pinv(
                    kernel_2, self.inv_iterations, self.pinverse_original_init
                )
                if self.use_razavi_pinverse
                else torch.linalg.pinv(kernel_2)
            )

            x = torch.matmul(
                torch.matmul(
                    kernel_1,
                    kernel_2_inv,
                ),
                kernel_3,
            )

        if self.skip_connection:
            v_conv = self.skip_connection(
                v.reshape(-1, self.num_heads, v.size(-2), v.size(-1))
            )
            x += v_conv.reshape(-1, v_conv.size(-2), v_conv.size(-1))
        x = self.attn_drop(x)
        return x

    def _triu_mask(
        self, dim_1: int, dim_2: int, dim_3: int, **kwargs
    ) -> Tensor:
        device = kwargs["device"]
        dtype = kwargs["dtype"]

        return torch.triu(
            torch.ones(dim_2, dim_3, dtype=dtype, device=device)
            * float("-inf"),
            diagonal=1,
        ).expand(dim_1, -1, -1)
