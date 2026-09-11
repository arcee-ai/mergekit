# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Convenience APIs for applying merge methods to in-memory objects."""

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Dict, Optional, Union

import torch

from mergekit.merge_methods.base import (
    BatchOptions,
    MergeMethod,
    PerGroupValues,
    TensorGroup,
)
from mergekit.merge_methods.buffers import copy_non_floating_buffer

StateDict = Mapping[str, torch.Tensor]
StateDictLike = Union[StateDict, torch.nn.Module]


def merge_tensors(
    tensors: Sequence[torch.Tensor],
    method: Union[str, MergeMethod],
    *,
    parameters: Optional[Mapping[str, Any]] = None,
    ids: Optional[Sequence[Hashable]] = None,
    base_index: Optional[int] = None,
    name: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Merge inputs contributing to one output tensor.

    Per-input parameters accept sequences in input order, mappings keyed by ids,
    or broadcast scalars. The base index always refers to the input sequence.
    Inputs are borrowed and promoted unless dtype is supplied; out_dtype casts
    the result. This uses the same validation and execution as a direct method call.
    Checkpoint buffer handling belongs to merge_state_dicts.
    """
    if isinstance(method, str):
        from mergekit import merge_methods

        method = merge_methods.get(method)
    group = TensorGroup.from_tensors(tensors, ids=ids, base_index=base_index, name=name)
    (result,) = method(
        (group,), parameters=parameters, dtype=dtype, out_dtype=out_dtype
    )
    return result


def merge_state_dicts(
    models: Union[Mapping[Hashable, StateDictLike], Sequence[StateDictLike]],
    method: Union[str, MergeMethod],
    *,
    parameters: Optional[Mapping[str, Any]] = None,
    base: Optional[Hashable] = None,
    strict: bool = True,
    dtype: Optional[torch.dtype] = None,
    out_dtype: Optional[torch.dtype] = None,
    batch_options: Optional[BatchOptions] = None,
) -> Dict[str, torch.Tensor]:
    """Merge in-memory modules or state dictionaries.

    Parameters are broadcast across tensor groups. Per-input parameters may be
    sequences in input order or mappings keyed by the input IDs. Compatible
    floating-point weights are batched up to batch_options' packing limits.
    Non-floating buffers are copied only when every input agrees exactly.
    Floating inputs are promoted independently for each weight unless dtype
    explicitly selects their representation. Inputs are converted per execution
    chunk; out_dtype casts outputs before they are retained. Module parameters
    retain their autograd connections; use torch.no_grad() to avoid recording merge
    operations. Autograd graphs and outputs that alias inputs may extend the
    lifetime of conversion storage.
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

    method.validate_inputs([key for key, _ in state_dicts], base)
    copied = {}
    merge_names = []
    merge_indices = []
    for index, name in enumerate(tensor_names):
        tensors = [state_dict[name] for _, state_dict in state_dicts]
        buffer = copy_non_floating_buffer(tensors, name)
        if buffer is None:
            merge_names.append(name)
            merge_indices.append(index)
        else:
            copied[name] = buffer

    resolved_parameters = {}
    for name, value in (parameters or {}).items():
        if isinstance(value, PerGroupValues):
            if len(value.values) != len(tensor_names):
                raise ValueError(
                    f"Parameter {name} expects {len(tensor_names)} group values"
                )
            value = PerGroupValues([value.values[index] for index in merge_indices])
        resolved_parameters[name] = value

    input_ids = [input_id for input_id, _ in state_dicts]
    base_index = input_ids.index(base) if base is not None else None
    groups = tuple(
        TensorGroup.from_tensors(
            [state_dict[name] for _, state_dict in state_dicts],
            ids=input_ids,
            base_index=base_index,
            name=name,
        )
        for name in merge_names
    )
    merged = method(
        groups,
        parameters=resolved_parameters,
        dtype=dtype,
        out_dtype=out_dtype,
        batch_options=batch_options,
    )
    results = dict(zip(merge_names, merged))
    results.update(copied)
    return {name: results[name] for name in tensor_names}


def _as_state_dict(model: StateDictLike) -> StateDict:
    if isinstance(model, torch.nn.Module):
        model = model.state_dict(keep_vars=True)
    if not isinstance(model, Mapping):
        raise TypeError(f"Expected a module or state dict, got {type(model).__name__}")
    if not all(isinstance(name, str) for name in model):
        raise TypeError("State dictionary keys must be strings")
    if not all(isinstance(tensor, torch.Tensor) for tensor in model.values()):
        raise TypeError("State dictionary values must be tensors")
    return model
