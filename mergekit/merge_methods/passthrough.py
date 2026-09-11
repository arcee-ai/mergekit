# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import Optional

import torch

from mergekit.merge_methods.base import (
    InputContract,
    PerInput,
    TensorGroup,
)
from mergekit.merge_methods.easy_define import from_group_kernel


def _passthrough_merge(
    group: TensorGroup, scale: PerInput[Optional[float]] = None
) -> torch.Tensor:
    entry = group.entries[0]
    value = scale[entry.id]
    return entry.tensor if value is None else entry.tensor * value


passthrough_merge_method = from_group_kernel(
    _passthrough_merge,
    name="passthrough",
    pretty_name="Passthrough",
    uses_accelerator=False,
    contract=InputContract(min_inputs=1, max_inputs=1),
)
