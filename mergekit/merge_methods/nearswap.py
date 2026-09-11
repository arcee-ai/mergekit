# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import torch

from mergekit.merge_methods.base import (
    BasePolicy,
    InputContract,
    Shared,
    TensorGroup,
)
from mergekit.merge_methods.easy_define import merge_method


@merge_method(
    name="nearswap",
    pretty_name="NearSwap",
    reference_url="https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001",
    contract=InputContract(
        base=BasePolicy.REQUIRED,
        min_inputs=2,
        max_inputs=2,
        min_non_base=1,
        max_non_base=1,
    ),
)
def nearswap_merge(group: TensorGroup, t: Shared[float]) -> torch.Tensor:
    a = group.base.tensor
    b = group.non_base[0].tensor

    absdiff = torch.abs(a - b)
    weight = (t / absdiff.clamp(min=1e-6)).clamp(min=0, max=1)
    return weight * b + (1 - weight) * a
