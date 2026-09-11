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
from mergekit.merge_methods.easy_define import from_batch_kernel


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
    """
    dtype = torch.float64 if v0.dtype == torch.float64 else torch.float32
    a = v0.reshape(v0.shape[0], -1).to(dtype)
    b = v1.reshape(v1.shape[0], -1).to(dtype)
    norm_a = torch.linalg.vector_norm(a, dim=1, keepdim=True)
    norm_b = torch.linalg.vector_norm(b, dim=1, keepdim=True)
    unit_a = a / torch.where(norm_a > eps, norm_a, 1)
    unit_b = b / torch.where(norm_b > eps, norm_b, 1)
    dot = (unit_a * unit_b).sum(dim=1, keepdim=True).clamp(-1, 1)
    linear = dot.abs() > dot_threshold
    # Avoid singularities even on the unselected branch of torch.where, so
    # collinear inputs also have finite gradients.
    theta = torch.acos(torch.where(linear, 0, dot))
    t = t.reshape(-1, 1).to(dtype)
    spherical = (torch.sin((1 - t) * theta) * a + torch.sin(t * theta) * b) / torch.sin(
        theta
    )
    result = torch.where(linear, (1 - t) * a + t * b, spherical)
    return result.reshape(v0.shape).to(v0.dtype)


def _slerp_merge(
    batch: TensorBatch,
    t: Annotated[torch.Tensor, BatchParameter(float)],
) -> torch.Tensor:
    if not batch.tensors.is_floating_point():
        raise TypeError("SLERP requires floating-point tensors")
    base_index = batch.base_index
    if base_index is None or batch.tensors.shape[1] != 2:
        raise ValueError("SLERP requires a base and one other input")
    return slerp(t, batch.tensors[:, base_index], batch.tensors[:, 1 - base_index])


slerp_merge_method = from_batch_kernel(
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
