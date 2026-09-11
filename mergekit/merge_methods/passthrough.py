# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import Optional

import torch

from mergekit.merge_methods.base import (
    InputContract,
    PerInput,
    TensorGroup,
    method_from_function,
)


def _passthrough_merge(
    group: TensorGroup, scale: PerInput[Optional[float]] = None
) -> torch.Tensor:
    entry = group.entries[0]
    value = scale[entry.id]
    return entry.tensor if value is None else entry.tensor * value


passthrough_merge_method = method_from_function(
    _passthrough_merge,
    name="passthrough",
    pretty_name="Passthrough",
    contract=InputContract(min_inputs=1, max_inputs=1),
)
