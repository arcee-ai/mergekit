# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import List

import torch

from mergekit.merge_methods.easy_define import merge_method


@merge_method(
    name="nearswap",
    pretty_name="NearSwap",
    reference_url="https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001",
)
def nearswap_merge(
    tensors: List[torch.Tensor], base_tensor: torch.Tensor, t: float
) -> torch.Tensor:
    if not tensors:
        return base_tensor
    if len(tensors) != 1:
        raise RuntimeError(
            "NearSwap merge expects exactly two models, one base and one other"
        )
    a = base_tensor
    b = tensors[0]

    absdiff = torch.abs(a - b)
    weight = (t / absdiff.clamp(min=1e-6)).clamp(min=0, max=1)
    return weight * b + (1 - weight) * a
