from typing import Dict

import torch

from config import ConfigReader
from graph import TensorReference
from merge_methods.base import MergeMethod


class PassthroughMerge(MergeMethod):
    def __call__(
        self,
        parameter_name: str,
        input_tensors: Dict[TensorReference, torch.Tensor],
        config: ConfigReader,
        **_kwargs,
    ) -> torch.Tensor:
        if len(input_tensors) != 1:
            raise RuntimeError("Passthrough merge expects exactly one tensor")

        return list(input_tensors.values())[0]
