from enum import Enum
from typing import Callable, Dict, Union

import torch
from typing_extensions import TypeAlias

from .linear import LinearMergeOptions, linear_merge_tensors
from .slerp import SlerpMergeOptions, slerp_merge_tensors
from .ties import TiesMergeOptions, ties_merge_tensors

MergeFunction: TypeAlias = Callable[[Dict, str, Dict[str, torch.Tensor]], torch.Tensor]
MergeOptions: TypeAlias = Union[TiesMergeOptions, LinearMergeOptions, SlerpMergeOptions]


class MergeMethod(str, Enum):
    ties = "ties"
    linear = "linear"
    slerp = "slerp"


def get(method: MergeFunction) -> MergeFunction:
    if method == MergeMethod.ties:
        return ties_merge_tensors
    elif method == MergeMethod.linear:
        return linear_merge_tensors
    elif method == MergeMethod.slerp:
        return slerp_merge_tensors
    raise RuntimeError(f"Unimplemented merge method {method}")


__all__ = [
    "MergeFunction",
    "get",
    "TiesMergeOptions",
    "ties_merge_tensors",
    "LinearMergeOptions",
    "linear_merge_tensors",
    "SlerpMergeOptions",
    "slerp_merge_tensors",
]
