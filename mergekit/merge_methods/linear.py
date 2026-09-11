# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import torch
from typing_extensions import Annotated

from mergekit.merge_methods.base import (
    BatchParameter,
    InputContract,
    Option,
    ParameterScope,
    TensorBatch,
)
from mergekit.merge_methods.easy_define import from_batch_kernel


def _linear_merge(
    batch: TensorBatch,
    weight: Annotated[torch.Tensor, BatchParameter(float, ParameterScope.INPUT)],
    normalize: Option[bool] = True,
) -> torch.Tensor:
    tensors = batch.tensors
    if not tensors.is_floating_point():
        raise TypeError("Linear merging requires floating-point tensors")
    # Accumulate low-precision inputs in float32; keep float64 when requested.
    weights = weight.reshape(*weight.shape, *((1,) * (tensors.ndim - 2)))
    result = (weights * tensors).sum(dim=1)
    if normalize:
        result = result / weights.sum(dim=1)
    return result.to(tensors.dtype)


linear_merge = from_batch_kernel(
    _linear_merge,
    name="linear",
    pretty_name="Linear",
    reference_url="https://arxiv.org/abs/2203.05482",
    contract=InputContract(min_inputs=1),
    rectify_embeddings=True,
)
