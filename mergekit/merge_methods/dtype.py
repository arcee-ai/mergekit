# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Execution-time dtype alignment; kernels receive already aligned inputs."""

from dataclasses import replace
from typing import Optional

import torch

from mergekit.merge_methods.base import TensorGroup


def promoted_dtype(group: TensorGroup) -> torch.dtype:
    dtype = group.entries[0].tensor.dtype
    for entry in group.entries[1:]:
        dtype = torch.promote_types(dtype, entry.tensor.dtype)
    return dtype


def align_dtype(group: TensorGroup, dtype: Optional[torch.dtype] = None) -> TensorGroup:
    if not group.entries:
        return group
    if dtype is None:
        dtype = promoted_dtype(group)
    return replace(
        group,
        entries=tuple(
            replace(entry, tensor=entry.tensor.to(dtype=dtype))
            for entry in group.entries
        ),
    )
