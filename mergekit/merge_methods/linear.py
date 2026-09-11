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
    # Normalize the small coefficient array before rounding to the input dtype.
    # The matrix product avoids full-sized conversion and weighted-input buffers.
    if normalize:
        weight = weight / weight.sum(dim=1, keepdim=True)
    flat = tensors.reshape(*tensors.shape[:2], -1)
    result = torch.bmm(weight.to(tensors.dtype).unsqueeze(1), flat).squeeze(1)
    return result.reshape(tensors.shape[0], *tensors.shape[2:])


linear_merge = merge_method(
    _linear_merge,
    name="linear",
    pretty_name="Linear",
    reference_url="https://arxiv.org/abs/2203.05482",
    contract=InputContract(min_inputs=1),
)
