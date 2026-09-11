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
    # Contract the input axis without materializing a weighted copy of every
    # input. Low-precision inputs still need a float32 conversion buffer.
    flat = tensors.reshape(*tensors.shape[:2], -1).to(weight.dtype)
    result = torch.bmm(weight.unsqueeze(1), flat).squeeze(1)
    del flat
    if normalize:
        result = result / weight.sum(dim=1, keepdim=True)
    return result.reshape(tensors.shape[0], *tensors.shape[2:]).to(tensors.dtype)


linear_merge = from_batch_kernel(
    _linear_merge,
    name="linear",
    pretty_name="Linear",
    reference_url="https://arxiv.org/abs/2203.05482",
    contract=InputContract(min_inputs=1),
)
