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
    tensors = batch.tensors
    if not tensors.is_floating_point():
        raise TypeError("Linear merging requires floating-point tensors")
    # Accumulate into one wide output buffer without converting every input or
    # rounding coefficients to the input dtype. In particular, nearly cancelling
    # weights must not overflow float16 during normalization.
    dtype = torch.float64 if tensors.dtype == torch.float64 else torch.float32
    weight = weight.to(dtype)
    coefficient_shape = (tensors.shape[0],) + (1,) * (tensors.ndim - 2)
    result = torch.zeros_like(tensors[:, 0], dtype=dtype)
    for tensor, coefficient in zip(tensors.unbind(1), weight.unbind(1)):
        result.addcmul_(tensor, coefficient.reshape(coefficient_shape))
    if normalize:
        # The small reduction needs extra precision too: parallel float32 sums
        # can lose a small residual such as sum([1, -1, 1e-5]).
        denominator = weight.sum(dim=1, dtype=torch.float64).to(dtype)
        result.div_(denominator.reshape(coefficient_shape))
    return result.to(tensors.dtype)


linear_merge = merge_method(
    _linear_merge,
    name="linear",
    pretty_name="Linear",
    reference_url="https://arxiv.org/abs/2203.05482",
    contract=InputContract(min_inputs=1),
)
