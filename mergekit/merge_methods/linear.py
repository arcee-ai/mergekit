# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import torch
from typing_extensions import Annotated

from mergekit.merge_methods.base import (
    BatchParameter,
    ParameterScope,
    TensorBatch,
)
from mergekit.merge_methods.easy_define import merge_method


def _linear_merge(
    batch: TensorBatch,
    weight: Annotated[torch.Tensor, BatchParameter(float, ParameterScope.INPUT)],
    normalize: bool = True,
) -> torch.Tensor:
    first = batch.tensors[0]
    if not first.is_floating_point():
        raise TypeError("Linear merging requires floating-point tensors")
    dtype = torch.float64 if first.dtype == torch.float64 else torch.float32
    weight = weight.to(dtype)
    coefficient_shape = (first.shape[0],) + (1,) * (first.ndim - 1)
    if normalize:
        denominator = weight.sum(dim=1).reshape(coefficient_shape)
        if (denominator == 0).any():
            raise ValueError("Cannot normalize weights that sum to zero")
    result = torch.zeros_like(first, dtype=dtype)
    for tensor, coefficient in zip(batch.tensors, weight.unbind(1)):
        result.addcmul_(tensor, coefficient.reshape(coefficient_shape))
    if normalize:
        result.div_(denominator)
    return result.to(first.dtype)


linear_merge = merge_method(
    _linear_merge,
    name="linear",
    pretty_name="Linear",
    reference_url="https://arxiv.org/abs/2203.05482",
)
