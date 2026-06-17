# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import re
from collections.abc import Callable, Mapping
from typing import Optional, Union

import torch
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightRenaming

TensorSource = Union[torch.Tensor, Callable[[], torch.Tensor]]


def _natural_sort_key(value: str):
    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"([0-9]+)", value)
    )


def convert_checkpoint_tensors(
    model_type: Optional[str],
    source_tensors: Mapping[str, TensorSource],
    target_key: str,
) -> Optional[torch.Tensor]:
    """Convert checkpoint-layout tensors to a Transformers v5 model-layout tensor.

    Transformers v5 owns the conversion registry. This helper keeps mergekit's
    tensor loading code independent from the registry's stateful loader protocol.
    """
    if not model_type:
        return None

    transforms = get_checkpoint_conversion_mapping(model_type)
    if not transforms:
        return None

    available: dict[str, TensorSource] = dict(source_tensors)
    for transform in transforms:
        additions: dict[str, TensorSource] = {}
        for source_key in sorted(available, key=_natural_sort_key):
            renamed_key, source_pattern = transform.rename_source_key(source_key)
            if source_pattern is None:
                continue

            if isinstance(transform, WeightRenaming):
                additions[renamed_key] = available[source_key]
                continue

            if renamed_key == target_key:
                transform.add_tensor(
                    target_key=renamed_key,
                    source_key=source_key,
                    source_pattern=source_pattern,
                    future=available[source_key],
                )

        available.update(additions)

        if isinstance(transform, WeightRenaming):
            continue

        if target_key in transform.layer_targets:
            converted = transform.convert(target_key)
            if target_key in converted:
                return converted[target_key]
            available.update(converted)

    tensor = available.get(target_key)
    if callable(tensor):
        tensor = tensor()
    return tensor if isinstance(tensor, torch.Tensor) else None


def can_convert_checkpoint_keys(
    model_type: Optional[str],
    source_keys: set[str],
    target_key: str,
) -> bool:
    """Return whether source checkpoint keys can produce a target model key."""
    if target_key in source_keys:
        return True
    if not model_type:
        return False

    transforms = get_checkpoint_conversion_mapping(model_type)
    if not transforms:
        return False

    available = set(source_keys)
    for transform in transforms:
        additions = set()
        target_sources = set()
        for source_key in sorted(available, key=_natural_sort_key):
            renamed_key, source_pattern = transform.rename_source_key(source_key)
            if source_pattern is None:
                continue
            if isinstance(transform, WeightRenaming):
                additions.add(renamed_key)
            elif renamed_key == target_key:
                target_sources.add(source_pattern)

        available.update(additions)
        if target_key in available:
            return True
        if target_sources and set(transform.source_patterns).issubset(target_sources):
            return True

    return False


def has_checkpoint_conversion(model_type: Optional[str]) -> bool:
    return bool(model_type and get_checkpoint_conversion_mapping(model_type))
