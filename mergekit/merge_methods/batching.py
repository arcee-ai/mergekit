# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Compatibility bucketing and incremental packing for numerical batch kernels."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from typing_extensions import Annotated, get_args, get_origin

from mergekit.merge_methods.base import (
    BasePolicy,
    BatchOptions,
    InputParameterTarget,
    MergeBatch,
    MergeMethodSpec,
    ParameterScope,
    TensorBatch,
    TensorEntry,
)


def _coefficient_dtype(value_type: Any, tensor_dtype: torch.dtype) -> torch.dtype:
    while get_origin(value_type) is Annotated:
        value_type = get_args(value_type)[0]
    if value_type is bool:
        return torch.bool
    if value_type is int:
        return torch.int64
    if value_type is float:
        return torch.float64 if tensor_dtype == torch.float64 else torch.float32
    raise TypeError("Batched coefficients must be bool, int, or float scalars")


@dataclass
class _PreparedGroup:
    index: int
    entries: Tuple[TensorEntry, ...]
    tensors: Tuple[torch.Tensor, ...]
    parameters: Dict[str, Any]
    base_index: Optional[int]
    coefficient_dtypes: Dict[str, torch.dtype]
    packed_bytes: int


@dataclass
class PreparedBatch:
    """A lightweight packing recipe; construction does not allocate tensor buffers."""

    groups: List[_PreparedGroup]

    @property
    def indices(self) -> Tuple[int, ...]:
        return tuple(group.index for group in self.groups)

    def pack(self, spec: MergeMethodSpec) -> Tuple[TensorBatch, Dict[str, Any]]:
        first = self.groups[0]
        tensors = torch.stack([t for group in self.groups for t in group.tensors])
        tensors = tensors.reshape(
            len(self.groups), len(first.entries), *tensors.shape[1:]
        )
        kwargs = {}
        for parameter in spec.parameters:
            if not parameter.batch_tensor:
                kwargs[parameter.name] = first.parameters[parameter.name]
                continue
            values = []
            for group in self.groups:
                value = group.parameters[parameter.name]
                if parameter.scope == ParameterScope.INPUT:
                    entries = group.entries
                    if parameter.input_target == InputParameterTarget.NON_BASE:
                        entries = tuple(entry for entry in entries if not entry.is_base)
                    value = value.values_for(entries)
                values.append(value)
            kwargs[parameter.name] = torch.tensor(
                values,
                dtype=first.coefficient_dtypes[parameter.name],
                device=tensors.device,
            )
        return TensorBatch(tensors, base_index=first.base_index, owned=True), kwargs


def prepare_batches(
    batch: MergeBatch,
    parameters: List[Dict[str, Any]],
    spec: MergeMethodSpec,
    options: BatchOptions,
) -> List[PreparedBatch]:
    """Bucket aligned groups and validate execution options without tensor math.

    Base-aware methods get a canonical base-first layout. Remaining inputs retain
    their relative order; coefficient mappings are aligned to that same layout.
    """
    buckets = {}
    for index, (group, bound) in enumerate(zip(batch.groups, parameters)):
        entries = group.entries
        base_index = None
        if group.base is not None and spec.contract.base != BasePolicy.IGNORED:
            entries = (group.base, *group.non_base)
            base_index = 0
        tensors = tuple(entry.tensor for entry in entries)
        if not tensors:
            raise ValueError("Numerical batch kernels require at least one input")
        first = tensors[0]

        execution_options = []
        coefficient_dtypes = {}
        packed_bytes = sum(t.numel() * t.element_size() for t in tensors)
        for parameter in spec.parameters:
            value = bound[parameter.name]
            if parameter.batch_tensor:
                values = (
                    list(value.values())
                    if parameter.scope == ParameterScope.INPUT
                    else [value]
                )
                dtype = _coefficient_dtype(parameter.value_type, first.dtype)
                coefficient_dtypes[parameter.name] = dtype
                itemsize = {
                    torch.bool: 1,
                    torch.int64: 8,
                    torch.float32: 4,
                    torch.float64: 8,
                }[dtype]
                packed_bytes += len(values) * itemsize
            else:
                try:
                    hash(value)
                except TypeError as error:
                    raise TypeError(
                        f"Execution option {parameter.name} must be hashable"
                    ) from error
                execution_options.append((parameter.name, type(value), value))
        key = (
            first.shape,
            first.dtype,
            first.device,
            len(entries),
            base_index,
            tuple(coefficient_dtypes.items()),
            tuple(execution_options),
        )
        buckets.setdefault(key, []).append(
            _PreparedGroup(
                index,
                entries,
                tensors,
                bound,
                base_index,
                coefficient_dtypes,
                packed_bytes,
            )
        )

    result = []
    for groups in buckets.values():
        chunk = []
        size = 0
        for group in groups:
            if chunk and (
                size + group.packed_bytes > options.max_bytes
                or len(chunk) >= options.max_groups
            ):
                result.append(PreparedBatch(chunk))
                chunk, size = [], 0
            chunk.append(group)
            size += group.packed_bytes
        if chunk:
            result.append(PreparedBatch(chunk))
    return result
