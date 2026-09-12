# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Explicit input transformations, separate from numerical merge contracts."""

import logging
from dataclasses import replace

from mergekit.merge_methods.base import TensorGroup
from mergekit.vocabulary import vocabulary_views


def truncate_vocabulary(group: TensorGroup) -> TensorGroup:
    """Unsafely retain the common token-ID prefix of vocabulary-dependent inputs.

    Token IDs in the retained prefix MUST already have identical meanings. This
    does not inspect tokenizers, align tokens, or repair output model metadata.
    Returns borrowed views without mutating inputs; non-vocabulary shapes must match.
    Groups without a vocabulary axis are left unchanged for normal merge validation.
    """
    axis = group.metadata.vocabulary_axis
    if axis is None or not group.entries:
        return group
    views = vocabulary_views(group.tensors, axis, group.metadata.name)
    shapes = [tuple(entry.tensor.shape) for entry in group.entries]
    rows = min(view.shape[0] for view in views)
    if all(view.shape[0] == rows for view in views):
        return group
    retained_shape = list(shapes[0])
    retained_shape[axis] = rows
    logging.warning(
        "UNSAFE vocabulary truncation for %s: %s -> %s (vocabulary_axis=%d). "
        "Discarding trailing token slices; this assumes identical token IDs in the retained prefix and "
        "can corrupt the model otherwise. Use tokenizer configuration "
        "(e.g. tokenizer: {source: base}) instead.",
        group.metadata.name,
        shapes,
        tuple(retained_shape),
        axis,
    )
    return replace(
        group,
        entries=tuple(
            replace(entry, tensor=entry.tensor.narrow(axis, 0, rows))
            for entry in group.entries
        ),
    )
