# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import torch

from mergekit.merge_methods.base import (
    InputContract,
    PerInput,
    Shared,
    TensorGroup,
    method_from_function,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


def _linear_merge(
    group: TensorGroup,
    weight: PerInput[float],
    normalize: Shared[bool] = True,
) -> torch.Tensor:
    entries = group.entries
    tensors = [entry.tensor for entry in entries]
    weights = weight.values_for(entries)

    rectify_embed_sizes(group.metadata, tensors)
    unique_shapes = {tensor.shape for tensor in tensors}
    if len(unique_shapes) != 1:
        raise RuntimeError(
            f"Tensor size mismatch for {group.metadata.name}, "
            f"sizes: {list(unique_shapes)}"
        )

    stacked = torch.stack(tensors, dim=0)
    weight_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
    while len(weight_tensor.shape) < len(stacked.shape):
        weight_tensor.unsqueeze_(-1)

    result = (weight_tensor * stacked).sum(dim=0)
    if normalize:
        result = result / weight_tensor.sum(dim=0)
    return result


linear_merge = method_from_function(
    _linear_merge,
    name="linear",
    pretty_name="Linear",
    reference_url="https://arxiv.org/abs/2203.05482",
    contract=InputContract(min_inputs=1),
)
