from typing import Dict

import torch

from common import rectify_embed_sizes
from config import ConfigReader
from graph import TensorReference
from merge_methods.base import MergeMethod


class LinearMerge(MergeMethod):
    def __call__(
        self,
        parameter_name: str,
        input_tensors: Dict[TensorReference, torch.Tensor],
        config: ConfigReader,
        **_kwargs,
    ) -> torch.Tensor:
        keys = list(input_tensors.keys())

        tensors = [input_tensors[key] for key in keys]
        weights = [
            config.parameter("weight", model=key.model, required=True) for key in keys
        ]

        rectify_embed_sizes(parameter_name, tensors)

        unique_shapes = set(t.shape for t in tensors)
        if len(unique_shapes) != 1:
            raise RuntimeError(
                f"Tensor size mismatch for {parameter_name}, sizes: {list(unique_shapes)}"
            )

        tensors = torch.stack(tensors, dim=0)
        weights = torch.tensor(weights, dtype=tensors.dtype, device=tensors.device)
        while len(weights.shape) < len(tensors.shape):
            weights.unsqueeze_(-1)

        res = (weights * tensors).sum(dim=0)
        if config.parameter("normalize", default=True):
            res /= weights.sum(dim=0)

        return res
