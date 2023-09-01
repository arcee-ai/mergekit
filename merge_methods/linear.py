import logging
from typing import Any, Dict, Union

import torch
from pydantic import BaseModel

from common import ModelReference


class LinearMergeOptions(BaseModel):
    weights: Dict[ModelReference, float]
    normalize: bool = True
    dtype: Any


def linear_merge_tensors(
    options: Union[LinearMergeOptions, Dict],
    param_name: str,
    tensors: Dict[ModelReference, torch.Tensor],
) -> torch.Tensor:
    if isinstance(options, Dict):
        options = LinearMergeOptions(**options)

    model_names = list(tensors.keys())

    tensors = [tensors[key] for key in model_names]
    weights = [options.weights.get(key, 0.0) for key in model_names]

    if "lm_head" in param_name or "embed_tokens" in param_name:
        # special case - if lm_head.weight or embed_tokens.weight have a size
        # mismatch, take the largest common submatrix of all of them
        min_size = [None, None]
        for t in tensors:
            for idx in range(2):
                if min_size[idx] is None or t.shape[idx] < min_size[idx]:
                    min_size[idx] = t.shape[idx]

        if not all(t.shape == min_size for t in tensors):
            logging.warning(
                f"Using common submatrix of size {tuple(min_size)} for {param_name}"
            )
            for idx in range(len(tensors)):
                tensors[idx] = tensors[idx][: min_size[0], : min_size[1]]

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
