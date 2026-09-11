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
from mergekit.merge_methods.easy_define import merge_method


def _linear_merge(
    batch: TensorBatch,
    weight: Annotated[torch.Tensor, BatchParameter(float, ParameterScope.INPUT)],
    normalize: Option[bool] = True,
) -> torch.Tensor:
    first = batch.tensors[0]
    if not first.is_floating_point():
        raise TypeError("Linear merging requires floating-point tensors")
    weight = weight.to(torch.float64)
    coefficient_shape = (first.shape[0],) + (1,) * (first.ndim - 1)
    if normalize:
        denominator = weight.sum(dim=1).reshape(coefficient_shape)
        if (denominator == 0).any():
            raise ValueError("Cannot normalize weights that sum to zero")
    # One accumulator, with the same precision as the coefficient sum. Inputs
    # remain borrowed; no full-sized conversion of every input is retained.
    result = torch.zeros_like(first, dtype=torch.float64)
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
    contract=InputContract(min_inputs=1),
)
