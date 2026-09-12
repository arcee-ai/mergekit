# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Shared checkpoint-buffer handling for in-memory and graph adapters."""

from typing import Optional, Sequence

import torch


def copy_non_floating_buffer(
    tensors: Sequence[torch.Tensor], name: Optional[str] = None
) -> Optional[torch.Tensor]:
    """Copy an agreed buffer, or return None when all inputs are floating point."""
    if all(tensor.is_floating_point() for tensor in tensors):
        return None
    first = tensors[0]
    if any(
        tensor.dtype != first.dtype or not torch.equal(tensor, first)
        for tensor in tensors[1:]
    ):
        raise ValueError(
            f"Non-floating buffer {name!r} differs between inputs; resolve it explicitly before merging"
        )
    return first.clone()
