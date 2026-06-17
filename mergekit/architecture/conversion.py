# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import re
from collections.abc import Callable, Mapping
from typing import Optional, Union

import torch
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightRenaming

TensorSource = Union[torch.Tensor, Callable[[], torch.Tensor]]


class _TransformMatch:
    def __init__(
        self,
        source_key: str,
        renamed_key: str,
        source_pattern: str,
        source_value: TensorSource,
        wildcard_values: tuple[str, ...],
    ):
        self.source_key = source_key
        self.renamed_key = renamed_key
        self.source_pattern = source_pattern
        self.source_value = source_value
        self.wildcard_values = wildcard_values


def _natural_sort_key(value: str):
    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"([0-9]+)", value)
    )


def _iter_transform_matches(transform, available: Mapping[str, TensorSource]):
    for source_key in sorted(available, key=_natural_sort_key):
        renamed_key, source_pattern = transform.rename_source_key(source_key)
        if source_pattern is not None:
            yield _TransformMatch(
                source_key=source_key,
                renamed_key=renamed_key,
                source_pattern=source_pattern,
                source_value=available[source_key],
                wildcard_values=_wildcard_values(source_pattern, source_key),
            )


def _expanded_target_keys(renamed_key: str, target_patterns: list[str]) -> set[str]:
    if not target_patterns:
        return {renamed_key}
    first_target = target_patterns[0]
    if first_target not in renamed_key:
        return {renamed_key}
    prefix, _sep, suffix = renamed_key.partition(first_target)
    return {prefix + target_pattern + suffix for target_pattern in target_patterns}


def _wildcard_values(pattern: str, source_key: str) -> tuple[str, ...]:
    if "*" not in pattern:
        return ()
    escaped = re.escape(pattern).replace(r"\*", r"([^.]*)")
    match = re.search(escaped, source_key)
    if not match:
        return ()
    return match.groups()


def _has_complete_wildcard_group(wildcard_values: set[tuple[str, ...]]) -> bool:
    return wildcard_values == {()} or len(wildcard_values) > 1


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
        conversion_layers = set()
        for match in _iter_transform_matches(transform, available):
            if isinstance(transform, WeightRenaming):
                additions[match.renamed_key] = match.source_value
                continue

            if target_key in _expanded_target_keys(
                match.renamed_key, transform.target_patterns
            ):
                transform.add_tensor(
                    target_key=match.renamed_key,
                    source_key=match.source_key,
                    source_pattern=match.source_pattern,
                    future=match.source_value,
                )
                conversion_layers.add(match.renamed_key)

        available.update(additions)

        if isinstance(transform, WeightRenaming):
            continue

        for conversion_layer in conversion_layers:
            converted = transform.convert(conversion_layer)
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
        target_wildcards: dict[str, set[tuple[str, ...]]] = {}
        for match in _iter_transform_matches(
            transform, {source_key: source_key for source_key in available}
        ):
            if isinstance(transform, WeightRenaming):
                additions.add(match.renamed_key)
            elif target_key in _expanded_target_keys(
                match.renamed_key, transform.target_patterns
            ):
                target_wildcards.setdefault(match.source_pattern, set()).add(
                    match.wildcard_values
                )

        available.update(additions)
        if target_key in available:
            return True
        if target_wildcards and all(
            target_wildcards.get(source_pattern)
            for source_pattern in transform.source_patterns
        ):
            wildcard_sets = {
                frozenset(target_wildcards[source_pattern])
                for source_pattern in transform.source_patterns
            }
            if len(wildcard_sets) == 1:
                wildcard_values = next(iter(wildcard_sets))
                if _has_complete_wildcard_group(wildcard_values):
                    return True
                return False
            else:
                return False

        if target_wildcards and len(transform.source_patterns) == 1:
            wildcard_values = next(iter(target_wildcards.values()))
            if _has_complete_wildcard_group(wildcard_values):
                return True

            return False

    return False
