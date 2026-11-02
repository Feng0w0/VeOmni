# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""standard causal_conv1d vendored Triton-Ascend pair."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor

from .....registry import SavedState
from ...optional import optional_tensor, unused_like


@dataclass(frozen=True)
class _Meta:
    """Activation and which optional tensors were real."""

    activation: str | None
    has_bias: bool
    has_cu_seqlens: bool


@lru_cache(maxsize=16)
def _host_varlen_metadata(
    device: torch.device, dtype: torch.dtype, boundaries: tuple[int, ...], chunk_size: int
) -> tuple[Tensor, Tensor]:
    """Cache immutable host boundaries and chunk tables on the target device."""
    rows = [
        (seq, chunk)
        for seq, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
        for chunk in range((end - start + chunk_size - 1) // chunk_size)
    ]
    # Blocking uploads complete before publishing the cache entry, including
    # when another stream later reuses it. No device-to-host read is needed.
    cu = torch.tensor(boundaries, dtype=dtype).to(device)
    indices = torch.tensor(rows, dtype=dtype).reshape(-1, 2).to(device)
    return cu, indices


def forward(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    *,
    activation: str | None = "silu",
    seq_idx: Tensor | None = None,
    backend: str | None = None,
) -> tuple[Tensor, SavedState]:
    """NPU causal conv1d. *weight* is FLA ``[D, W]``; the kernel wants ``[W, D]``."""
    from ...vendor.triton.convolution import causal_conv1d_chunk_size, causal_conv1d_fwd_impl

    del seq_idx, backend
    if bias is None:
        bias = unused_like(weight)
    chunk_indices = unused_like(x, dtype=torch.int32)
    if cu_seqlens is None:
        cu_seqlens = unused_like(x, dtype=torch.int32)
    elif cu_seqlens.device.type == "cpu" and cu_seqlens.numel():
        cu_seqlens, chunk_indices = _host_varlen_metadata(
            x.device, cu_seqlens.dtype, tuple(cu_seqlens.tolist()), causal_conv1d_chunk_size(*x.shape[:2])
        )
        # Protect cached storage if eviction happens before this stream finishes.
        stream = torch.npu.current_stream(x.device)
        cu_seqlens.record_stream(stream)
        chunk_indices.record_stream(stream)
    elif cu_seqlens.device != x.device:
        cu_seqlens = cu_seqlens.to(device=x.device, non_blocking=True)

    weight_wd = weight.transpose(0, 1).contiguous()
    bias_opt = optional_tensor(bias)
    cu_opt = optional_tensor(cu_seqlens)
    output, _final_state = causal_conv1d_fwd_impl(
        x=x,
        weight=weight_wd,
        bias=bias_opt,
        residual=None,
        initial_state=None,
        activation=activation,
        cu_seqlens=cu_opt,
        output_final_state=False,
        chunk_indices=optional_tensor(chunk_indices),
    )
    return output, SavedState(
        (x, weight_wd, bias, cu_seqlens, chunk_indices),
        _Meta(activation, bias_opt is not None, cu_opt is not None),
    )


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, ...]:
    """Return ``(grad_x, grad_weight, grad_bias, grad_cu_seqlens)``."""
    from ...vendor.triton.convolution import causal_conv1d_bwd_impl

    meta = saved.metadata
    assert isinstance(meta, _Meta)
    x, weight_wd, bias, cu_seqlens, chunk_indices = saved.tensors
    bias_opt = bias if meta.has_bias else None
    cu_opt = cu_seqlens if meta.has_cu_seqlens else None
    if chunk_indices.numel():
        stream = torch.npu.current_stream(x.device)
        cu_seqlens.record_stream(stream)
        chunk_indices.record_stream(stream)
    grad_x, grad_weight_wd, grad_bias, _grad_residual, _grad_h0 = causal_conv1d_bwd_impl(
        x=x,
        dy=grad_output,
        dht=None,
        weight=weight_wd,
        bias=bias_opt,
        residual=None,
        initial_state=None,
        activation=meta.activation,
        cu_seqlens=cu_opt,
        chunk_indices=optional_tensor(chunk_indices),
    )
    grad_weight = None if grad_weight_wd is None else grad_weight_wd.transpose(0, 1)
    if not meta.has_bias:
        grad_bias = None
    return grad_x, grad_weight, grad_bias, None
