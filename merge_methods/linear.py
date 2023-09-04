from typing import Dict, Union

import torch
from pydantic import BaseModel

from common import ModelReference, rectify_embed_sizes


class LinearMergeOptions(BaseModel):
    weight: Dict[ModelReference, float]
    normalize: bool = True


def linear_merge_tensors(
    options: Union[LinearMergeOptions, Dict],
    param_name: str,
    tensors: Dict[ModelReference, torch.Tensor],
) -> torch.Tensor:
    if isinstance(options, Dict):
        options = LinearMergeOptions(**options)

    model_names = list(tensors.keys())

    tensors = [tensors[key] for key in model_names]
    weights = [options.weight.get(key, 0.0) for key in model_names]

    rectify_embed_sizes(param_name, tensors)

    unique_shapes = set(t.shape for t in tensors)
    if len(unique_shapes) != 1:
        raise RuntimeError(
            f"Tensor size mismatch for {param_name}, sizes: {list(unique_shapes)}"
        )

    tensors = torch.stack(tensors, dim=0)
    weights = torch.tensor(weights, dtype=tensors.dtype, device=tensors.device)
    while len(weights.shape) < len(tensors.shape):
        weights.unsqueeze_(-1)

    res = (weights * tensors).sum(dim=0)
    if options.normalize:
        res /= weights.sum(dim=0)

    return res
