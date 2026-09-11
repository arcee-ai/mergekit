# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import torch
from typing_extensions import Annotated

from mergekit.merge_methods.base import (
    BasePolicy,
    BatchParameter,
    InputContract,
    OptionalTensorPolicy,
    TensorBatch,
)
from mergekit.merge_methods.easy_define import merge_method

# Bound full-precision scratch independently of the size of a logical weight.
_CHUNK_ELEMENTS = 1024 * 1024


def slerp(
    t: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    dot_threshold: float = 0.9995,
    eps: float = 1e-8,
) -> torch.Tensor:
    """SLERP over weight dimensions, preserving the leading output-batch axis.

    All computation stays on the input device. Collinear and antipodal pairs use
    linear interpolation independently for each output, including in mixed batches.
    Work in float32 unless the inputs are float64. Chunking bounds inference
    scratch, not source/output storage or autograd graphs.
    """
    dtype = torch.float64 if v0.dtype == torch.float64 else torch.float32
    a = v0.reshape(v0.shape[0], -1)
    b = v1.reshape(v1.shape[0], -1)
    t = t.reshape(-1, 1).to(dtype)
    if not a.shape[1]:
        return ((1 - t) * a + t * b).reshape(v0.shape).to(v0.dtype)

    chunk_size = max(1, _CHUNK_ELEMENTS // a.shape[0])
    norm_a_parts, norm_b_parts = [], []
    for start in range(0, a.shape[1], chunk_size):
        x = a[:, start : start + chunk_size].to(dtype)
        y = b[:, start : start + chunk_size].to(dtype)
        norm_a_parts.append(torch.linalg.vector_norm(x, dim=1, keepdim=True))
        norm_b_parts.append(torch.linalg.vector_norm(y, dim=1, keepdim=True))
        del x, y

    # Norms of chunk norms preserve the zero-vector derivative, unlike sqrt of
    # a sum of squares. All retained statistics are small [chunks, outputs, 1].
    norm_a = torch.linalg.vector_norm(torch.stack(norm_a_parts), dim=0)
    norm_b = torch.linalg.vector_norm(torch.stack(norm_b_parts), dim=0)
    norm_a = torch.where(norm_a > eps, norm_a, 1)
    norm_b = torch.where(norm_b > eps, norm_b, 1)
    dot_parts = []
    for start in range(0, a.shape[1], chunk_size):
        # Normalize before multiplication to avoid overflowing the raw dot.
        x = a[:, start : start + chunk_size].to(dtype) / norm_a
        y = b[:, start : start + chunk_size].to(dtype) / norm_b
        dot_parts.append((x * y).sum(dim=1, keepdim=True))
        del x, y
    dot = torch.stack(dot_parts).sum(dim=0).clamp(-1, 1)
    linear = dot.abs() > dot_threshold
    # Avoid singularities even on the unselected branch of torch.where, so
    # collinear inputs also have finite gradients.
    theta = torch.acos(torch.where(linear, 0, dot))
    sin_theta = torch.sin(theta)
    coef_a = torch.where(linear, 1 - t, torch.sin((1 - t) * theta) / sin_theta)
    coef_b = torch.where(linear, t, torch.sin(t * theta) / sin_theta)

    result = torch.empty_like(a)
    for start in range(0, a.shape[1], chunk_size):
        x = a[:, start : start + chunk_size].to(dtype)
        y = b[:, start : start + chunk_size].to(dtype)
        result[:, start : start + chunk_size] = coef_a * x + coef_b * y
        del x, y
    return result.reshape(v0.shape)


def _slerp_merge(
    batch: TensorBatch,
    t: Annotated[torch.Tensor, BatchParameter(float)],
) -> torch.Tensor:
    if not batch.tensors[0].is_floating_point():
        raise TypeError("SLERP requires floating-point tensors")
    base_index = batch.base_index
    if base_index is None or len(batch.tensors) != 2:
        raise ValueError("SLERP requires a base and one other input")
    return slerp(t, batch.tensors[base_index], batch.tensors[1 - base_index])


slerp_merge_method = merge_method(
    _slerp_merge,
    name="slerp",
    pretty_name="SLERP",
    reference_url="https://en.wikipedia.org/wiki/Slerp",
    optional_tensor_policy=OptionalTensorPolicy.PASSTHROUGH_SINGLETON,
    contract=InputContract(
        base=BasePolicy.REQUIRED,
        min_inputs=2,
        max_inputs=2,
        min_non_base=1,
        max_non_base=1,
    ),
)
