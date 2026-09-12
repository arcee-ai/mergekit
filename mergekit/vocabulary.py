# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Layout validation shared by vocabulary alignment and truncation."""

from typing import Optional, Sequence, Tuple

import torch


def vocabulary_views(
    tensors: Sequence[torch.Tensor], axis: int, name: Optional[str] = None
) -> Tuple[torch.Tensor, ...]:
    """Borrow views with vocabulary first, requiring matching non-vocabulary shapes."""
    shapes = [tuple(tensor.shape) for tensor in tensors]
    if type(axis) is not int or axis < 0 or any(axis >= t.ndim for t in tensors):
        raise ValueError(f"Invalid vocabulary_axis={axis} for {name}: {shapes}")
    views = tuple(tensor.movedim(axis, 0) for tensor in tensors)
    if any(view.shape[0] == 0 for view in views):
        raise ValueError(f"Empty vocabulary for {name}: {shapes}")
    if views and any(view.shape[1:] != views[0].shape[1:] for view in views):
        raise ValueError(
            f"Non-vocabulary dimensions must match for {name} "
            f"(vocabulary_axis={axis}): {shapes}"
        )
    return views
