from typing import Callable, Dict

import torch
from .linear import LinearMergeOptions, linear_merge_tensors
from .ties import TiesMergeOptions, ties_merge_tensors
from typing_extensions import TypeAlias

MergeFunction: TypeAlias = Callable[[Dict, str, Dict[str, torch.Tensor]], torch.Tensor]


def get(name: str) -> MergeFunction:
    if name == "linear":
        return linear_merge_tensors
    elif name == "ties":
        return ties_merge_tensors
    raise RuntimeError(f"Unimplemented merge method '{name}'")


__all__ = [
    "MergeFunction",
    "get",
    "TiesMergeOptions",
    "ties_merge_tensors",
    "LinearMergeOptions",
    "linear_merge_tensors",
]
