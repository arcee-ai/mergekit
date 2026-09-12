# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Compatibility bucketing and incremental packing for numerical batch kernels."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from typing_extensions import Annotated, get_args, get_origin

from mergekit.merge_methods.base import (
    BasePolicy,
    BatchOptions,
    MergeMethodSpec,
    ParameterScope,
    TensorBatch,
    TensorEntry,
    TensorGroup,
)


def _coefficient_dtype(value_type: Any, input_dtype: torch.dtype) -> torch.dtype:
    while get_origin(value_type) is Annotated:
        value_type = get_args(value_type)[0]
    if value_type is bool:
        return torch.bool
    if value_type is int:
        return torch.int64
    if value_type is float:
        return torch.float64 if input_dtype == torch.float64 else torch.float32
    raise TypeError("Batched coefficients must be bool, int, or float scalars")


@dataclass
class _PreparedGroup:
    index: int
    entries: Tuple[TensorEntry, ...]
    tensors: Tuple[torch.Tensor, ...]
    parameters: Dict[str, Any]
    base_index: Optional[int]
    packed_bytes: int
    dtype: torch.dtype


@dataclass
class PreparedBatch:
    """A lightweight packing recipe; construction does not allocate tensor buffers."""

    groups: List[_PreparedGroup]

    @property
    def indices(self) -> Tuple[int, ...]:
        return tuple(group.index for group in self.groups)

    def pack(self, spec: MergeMethodSpec) -> Tuple[TensorBatch, Dict[str, Any]]:
        first = self.groups[0]
        tensors = []
        for sources in zip(*(group.tensors for group in self.groups)):
            if len(sources) == 1:
                # Borrow even strided singleton inputs; only dtype conversion copies.
                tensor = sources[0].to(dtype=first.dtype).unsqueeze(0)
            elif all(t.dtype == first.dtype for t in sources):
                tensor = torch.stack(sources)
            else:
                # Copy into the target dtype without separate conversion buffers.
                tensor = torch.empty(
                    (len(sources), *sources[0].shape),
                    dtype=first.dtype,
                    device=sources[0].device,
                )
                for index, source in enumerate(sources):
                    tensor[index].copy_(source)
            tensors.append(tensor)
        kwargs = {}
        for parameter in spec.parameters:
            if not parameter.batch_tensor:
                kwargs[parameter.name] = first.parameters[parameter.name]
                continue
            values = []
            for group in self.groups:
                value = group.parameters[parameter.name]
                if parameter.scope != ParameterScope.SHARED:
                    entries = group.entries
                    if parameter.scope == ParameterScope.NON_BASE:
                        entries = tuple(entry for entry in entries if not entry.is_base)
                    value = value.values_for(entries)
                values.append(value)
            kwargs[parameter.name] = torch.tensor(
                values,
                dtype=_coefficient_dtype(parameter.value_type, first.dtype),
                device=tensors[0].device,
            )
        return TensorBatch(tuple(tensors), base_index=first.base_index), kwargs


def prepare_batches(
    groups: Sequence[TensorGroup],
    parameters: List[Dict[str, Any]],
    spec: MergeMethodSpec,
    options: BatchOptions,
    *,
    input_dtypes: Sequence[Optional[torch.dtype]],
) -> List[PreparedBatch]:
    """Bucket compatible groups and validate execution options without tensor math.

    Base-aware methods get a canonical base-first layout. Remaining inputs retain
    their relative order; coefficient mappings are aligned to that same layout.
    Input dtypes specify conversion at packing time, and determine both
    compatibility and buffer sizes without allocating converted source tensors.
    """
    buckets = {}
    for index, (group, bound) in enumerate(zip(groups, parameters)):
        entries = group.entries
        base_index = None
        if group.base is not None and spec.contract.base != BasePolicy.IGNORED:
            entries = (group.base, *group.non_base)
            base_index = 0
        tensors = tuple(entry.tensor for entry in entries)
        if not tensors:
            raise ValueError("Numerical batch kernels require at least one input")
        first = tensors[0]
        target_dtype = input_dtypes[index] or first.dtype

        execution_options = []
        packed_bytes = sum(t.numel() for t in tensors) * target_dtype.itemsize
        for parameter in spec.parameters:
            value = bound[parameter.name]
            if parameter.batch_tensor:
                values = (
                    list(value.values())
                    if parameter.scope != ParameterScope.SHARED
                    else [value]
                )
                coefficient_dtype = _coefficient_dtype(
                    parameter.value_type, target_dtype
                )
                if coefficient_dtype == torch.int64:
                    limits = torch.iinfo(coefficient_dtype)
                    if any(
                        value < limits.min or value > limits.max for value in values
                    ):
                        raise ValueError(
                            f"Parameter {parameter.name} must fit in torch.int64"
                        )
                packed_bytes += len(values) * coefficient_dtype.itemsize
            else:
                try:
                    hash(value)
                except TypeError as error:
                    raise TypeError(
                        f"Execution option {parameter.name} must be hashable"
                    ) from error
                execution_options.append((parameter.name, type(value), value))
        prepared = _PreparedGroup(
            index,
            entries,
            tensors,
            bound,
            base_index,
            packed_bytes,
            target_dtype,
        )
        # There is nothing to bucket or partition for a singleton. It can use the
        # common packer immediately, including when it exceeds the packing budget.
        if len(groups) == 1:
            return [PreparedBatch([prepared])]
        key = (
            first.shape,
            target_dtype,
            first.device,
            len(entries),
            base_index,
            tuple(execution_options),
        )
        buckets.setdefault(key, []).append(prepared)

    result = []
    for bucket in buckets.values():
        chunk = []
        size = 0
        for group in bucket:
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
