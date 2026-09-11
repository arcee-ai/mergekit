# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Convenience APIs for applying merge methods to in-memory objects."""

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Dict, Optional, Union

import torch

from mergekit.merge_methods.base import (
    MergeBatch,
    MergeMethod,
    TensorEntry,
    TensorGroup,
    TensorMetadata,
)

StateDict = Mapping[str, torch.Tensor]
StateDictLike = Union[StateDict, torch.nn.Module]


def merge_state_dicts(
    models: Union[Mapping[Hashable, StateDictLike], Sequence[StateDictLike]],
    method: Union[str, MergeMethod],
    *,
    parameters: Optional[Mapping[str, Any]] = None,
    base: Optional[Hashable] = None,
    strict: bool = True,
) -> Dict[str, torch.Tensor]:
    """Merge in-memory modules or state dictionaries.

    Parameters are broadcast across tensor groups. Per-input parameters may be
    sequences in input order or mappings keyed by the input IDs.
    """

    if isinstance(method, str):
        from mergekit import merge_methods

        method = merge_methods.get(method)

    if isinstance(models, Mapping):
        items = list(models.items())
    else:
        items = list(enumerate(models))
    if not items:
        raise ValueError("At least one model is required")

    state_dicts = [(_id, _as_state_dict(model)) for _id, model in items]
    if len({key for key, _ in state_dicts}) != len(state_dicts):
        raise ValueError("Duplicate model IDs")
    if base is not None and base not in {key for key, _ in state_dicts}:
        raise ValueError(f"Unknown base input: {base!r}")

    key_sets = [set(state_dict) for _, state_dict in state_dicts]
    if strict and any(keys != key_sets[0] for keys in key_sets[1:]):
        missing = {
            key: sorted(set.union(*key_sets) - keys)
            for (key, _), keys in zip(state_dicts, key_sets)
            if keys != set.union(*key_sets)
        }
        raise ValueError(f"State dictionaries have different tensor keys: {missing}")
    tensor_names = (
        list(state_dicts[0][1])
        if strict
        else [
            name for name in state_dicts[0][1] if all(name in keys for keys in key_sets)
        ]
    )

    groups = tuple(
        TensorGroup(
            entries=tuple(
                TensorEntry(
                    id=input_id,
                    tensor=state_dict[name],
                    is_base=input_id == base,
                )
                for input_id, state_dict in state_dicts
            ),
            metadata=TensorMetadata(name=name),
        )
        for name in tensor_names
    )
    merged = method(MergeBatch(groups=groups), **dict(parameters or {}))
    return dict(zip(tensor_names, merged.tensors))


def _as_state_dict(model: StateDictLike) -> StateDict:
    if isinstance(model, torch.nn.Module):
        return model.state_dict()
    if not isinstance(model, Mapping):
        raise TypeError(f"Expected a module or state dict, got {type(model).__name__}")
    if not all(isinstance(name, str) for name in model):
        raise TypeError("State dictionary keys must be strings")
    if not all(isinstance(tensor, torch.Tensor) for tensor in model.values()):
        raise TypeError("State dictionary values must be tensors")
    return model
