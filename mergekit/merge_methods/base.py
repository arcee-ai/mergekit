# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Consumer-neutral interfaces for tensor merge methods.

Merge methods operate on in-memory tensors. Configuration readers, model loaders, and
the computation graph are adapters around this module rather than part of the method
interface itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import torch
from pydantic import TypeAdapter
from typing_extensions import Annotated, TypeAlias


class ParameterScope(str, Enum):
    """The input axis along which a parameter is bound."""

    SHARED = "shared"
    INPUT = "input"


class InputParameterTarget(str, Enum):
    ALL = "all"
    NON_BASE = "non_base"


@dataclass(frozen=True)
class ParameterMarker:
    scope: ParameterScope
    target: InputParameterTarget = InputParameterTarget.ALL


@dataclass(frozen=True)
class BatchParameter:
    """Annotate a kernel Tensor argument with its logical scalar type and axes."""

    value_type: Any
    scope: ParameterScope = ParameterScope.SHARED
    target: InputParameterTarget = InputParameterTarget.ALL


@dataclass(frozen=True)
class OptionMarker:
    """A Python-valued execution option, constant within each packed batch."""


T = TypeVar("T")


class PerInputValues(Mapping[Hashable, T], Generic[T]):
    """Immutable values associated with tensor inputs, preserving input order."""

    def __init__(self, items: Sequence[Tuple[Hashable, T]]):
        self._items = tuple(items)
        self._data = dict(self._items)
        if len(self._data) != len(self._items):
            raise ValueError("Duplicate input IDs in per-input parameter")

    def __getitem__(self, key: Hashable) -> T:
        return self._data[key]

    def __iter__(self) -> Iterator[Hashable]:
        return iter(key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def values_for(self, entries: Sequence["TensorEntry"]) -> List[T]:
        return [self[entry.id] for entry in entries]


@dataclass(frozen=True)
class PerGroupValues(Generic[T]):
    """Explicit values along the outer MergeBatch axis."""

    values: Tuple[T, ...]

    def __init__(self, values: Sequence[T]):
        object.__setattr__(self, "values", tuple(values))


# These annotations are both documentation and the source of ParameterSpec.scope.
# At runtime Shared[T] is T and PerInput[T] is PerInputValues[T].
Shared: TypeAlias = Annotated[T, ParameterMarker(ParameterScope.SHARED)]
Option: TypeAlias = Annotated[T, OptionMarker()]
PerInput: TypeAlias = Annotated[
    PerInputValues[T], ParameterMarker(ParameterScope.INPUT)
]
PerNonBase: TypeAlias = Annotated[
    PerInputValues[T],
    ParameterMarker(ParameterScope.INPUT, InputParameterTarget.NON_BASE),
]


MISSING = object()


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    value_type: Any
    scope: ParameterScope
    input_target: InputParameterTarget = InputParameterTarget.ALL
    default: Any = MISSING
    description: Optional[str] = None
    batch_tensor: bool = False
    _adapter: TypeAdapter = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_adapter", TypeAdapter(self.value_type))

    @property
    def required(self) -> bool:
        return self.default is MISSING

    def validate(self, value: Any) -> Any:
        return self._adapter.validate_python(value)


class BasePolicy(str, Enum):
    OPTIONAL = "optional"
    REQUIRED = "required"
    FORBIDDEN = "forbidden"
    IGNORED = "ignored"


class OptionalTensorPolicy(str, Enum):
    """Graph-adapter handling of optional weights missing from some inputs."""

    ERROR = "error"
    PASSTHROUGH_SINGLETON = "passthrough_singleton"
    BASE_OR_SKIP = "base_or_skip"


@dataclass(frozen=True)
class InputContract:
    """Declarative structural requirements for each tensor group."""

    base: BasePolicy = BasePolicy.IGNORED
    min_inputs: int = 1
    max_inputs: Optional[int] = None
    min_non_base: Optional[int] = None
    max_non_base: Optional[int] = None

    def __post_init__(self):
        for name in ("min_inputs", "min_non_base"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        for minimum_name, maximum_name in (
            ("min_inputs", "max_inputs"),
            ("min_non_base", "max_non_base"),
        ):
            minimum = getattr(self, minimum_name)
            maximum = getattr(self, maximum_name)
            if maximum is not None and maximum < (minimum or 0):
                raise ValueError(f"{maximum_name} cannot be less than {minimum_name}")

    def validate_ids(
        self,
        input_ids: Sequence[Hashable],
        base_id: Optional[Hashable] = None,
        *,
        group_name: Optional[str] = None,
    ) -> None:
        label = f" for {group_name}" if group_name else ""
        if any(input_id is None for input_id in input_ids):
            raise ValueError("Merge input IDs cannot be None")
        if len(set(input_ids)) != len(input_ids):
            raise ValueError(f"Duplicate merge inputs{label}")

        if self.base == BasePolicy.IGNORED:
            base_id = None
        has_base = base_id is not None and base_id in input_ids
        if base_id is not None and not has_base:
            raise ValueError(f"Base input is not present in merge inputs{label}")
        if self.base == BasePolicy.REQUIRED and not has_base:
            raise ValueError(f"Merge method requires a base input{label}")
        if self.base == BasePolicy.FORBIDDEN and has_base:
            raise ValueError(f"Merge method does not accept a base input{label}")

        count = len(input_ids)
        non_base_count = count - int(has_base)
        self._validate_count("inputs", count, self.min_inputs, self.max_inputs, label)
        if self.min_non_base is not None or self.max_non_base is not None:
            self._validate_count(
                "non-base inputs",
                non_base_count,
                self.min_non_base or 0,
                self.max_non_base,
                label,
            )

    @staticmethod
    def _validate_count(
        what: str,
        count: int,
        minimum: int,
        maximum: Optional[int],
        label: str,
    ) -> None:
        if count < minimum:
            raise ValueError(
                f"Merge method requires at least {minimum} {what}{label}; got {count}"
            )
        if maximum is not None and count > maximum:
            expectation = (
                f"exactly {minimum}" if minimum == maximum else f"at most {maximum}"
            )
            raise ValueError(
                f"Merge method requires {expectation} {what}{label}; got {count}"
            )


@dataclass(frozen=True)
class TensorMetadata:
    name: Optional[str] = None
    is_embed: bool = False
    optional: bool = False

    @classmethod
    def from_weight_info(cls, weight: Any) -> "TensorMetadata":
        return cls(name=weight.name, is_embed=weight.is_embed, optional=weight.optional)


@dataclass(frozen=True)
class TensorEntry:
    id: Hashable
    tensor: torch.Tensor
    is_base: bool = False

    def __post_init__(self):
        if self.id is None:
            raise ValueError("Merge input IDs cannot be None")
        try:
            hash(self.id)
        except TypeError as error:
            raise TypeError("Tensor entry IDs must be hashable") from error
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("TensorEntry.tensor must be a torch.Tensor")


@dataclass(frozen=True)
class TensorGroup:
    """Inputs which contribute to one logical output tensor."""

    entries: Tuple[TensorEntry, ...]
    metadata: TensorMetadata = field(default_factory=TensorMetadata)

    def __post_init__(self):
        object.__setattr__(self, "entries", tuple(self.entries))
        if len({entry.id for entry in self.entries}) != len(self.entries):
            raise ValueError("Duplicate tensor input IDs")
        if sum(entry.is_base for entry in self.entries) > 1:
            raise ValueError("A tensor group can have at most one base input")

    @property
    def base(self) -> Optional[TensorEntry]:
        return next((entry for entry in self.entries if entry.is_base), None)

    @property
    def non_base(self) -> Tuple[TensorEntry, ...]:
        return tuple(entry for entry in self.entries if not entry.is_base)

    @property
    def tensors(self) -> Tuple[torch.Tensor, ...]:
        return tuple(entry.tensor for entry in self.entries)

    def validate_tensors(self) -> None:
        """Require aligned inputs without allocating or changing their storage."""
        if not self.entries:
            return
        first = self.entries[0].tensor
        if any(entry.tensor.shape != first.shape for entry in self.entries):
            hint = ""
            if self.metadata.is_embed:
                hint = (
                    " Align vocabulary rows with tokenizer configuration before "
                    "merging (for example, tokenizer: {source: base}); hidden "
                    "dimensions must already match. Direct callers must align "
                    "their tensors themselves."
                )
            raise ValueError(
                f"Tensor size mismatch for {self.metadata.name}: "
                f"{[tuple(entry.tensor.shape) for entry in self.entries]}.{hint}"
            )
        if any(
            entry.tensor.dtype != first.dtype or entry.tensor.device != first.device
            for entry in self.entries
        ):
            raise ValueError(
                f"Inputs for {self.metadata.name} must have the same dtype and device"
            )


@dataclass(frozen=True)
class MergeBatch:
    groups: Tuple[TensorGroup, ...]

    def __post_init__(self):
        object.__setattr__(self, "groups", tuple(self.groups))
        if not all(isinstance(group, TensorGroup) for group in self.groups):
            raise TypeError("MergeBatch.groups must contain TensorGroup values")

    @classmethod
    def from_tensors(
        cls,
        tensors: Sequence[torch.Tensor],
        *,
        ids: Optional[Sequence[Hashable]] = None,
        base_index: Optional[int] = None,
        name: Optional[str] = None,
    ) -> "MergeBatch":
        ids = tuple(range(len(tensors))) if ids is None else tuple(ids)
        if len(ids) != len(tensors):
            raise ValueError("ids and tensors must have the same length")
        if base_index is not None and not 0 <= base_index < len(tensors):
            raise ValueError("base_index is out of range")
        entries = tuple(
            TensorEntry(id=key, tensor=tensor, is_base=idx == base_index)
            for idx, (key, tensor) in enumerate(zip(ids, tensors))
        )
        return cls(
            groups=(TensorGroup(entries=entries, metadata=TensorMetadata(name=name)),)
        )


@dataclass(frozen=True)
class MergedBatch:
    tensors: Tuple[torch.Tensor, ...]

    def __post_init__(self):
        object.__setattr__(self, "tensors", tuple(self.tensors))

    def one(self) -> torch.Tensor:
        if len(self.tensors) != 1:
            raise ValueError(f"Expected one merged tensor, got {len(self.tensors)}")
        return self.tensors[0]


@dataclass(frozen=True)
class BatchOptions:
    """Limits on packing, not total device memory or kernel scratch space.

    A single oversized group executes alone. Source tensors and returned outputs
    are not included in max_bytes. max_groups also bounds batches of empty tensors.
    """

    max_bytes: int = 64 * 1024 * 1024
    max_groups: int = 256

    def __post_init__(self):
        if self.max_bytes <= 0 or self.max_groups <= 0:
            raise ValueError("Batch limits must be positive")


@dataclass(frozen=True)
class TensorBatch:
    """Numerical kernel input with axes [output, input, *weight_shape].

    Source tensors are borrowed by default. The preparer supplies owned packing
    buffers; kernels may obtain writable storage through workspace(). No model
    IDs or configuration objects cross this boundary.
    """

    tensors: torch.Tensor
    base_index: Optional[int] = None
    owned: bool = False

    def __post_init__(self):
        if self.tensors.ndim < 2 or min(self.tensors.shape[:2]) < 1:
            raise ValueError("TensorBatch requires nonempty output and input axes")
        if (
            self.base_index is not None
            and not 0 <= self.base_index < self.tensors.shape[1]
        ):
            raise ValueError("base_index is out of range")

    def workspace(self) -> torch.Tensor:
        """Return writable storage without modifying borrowed source tensors."""
        return self.tensors if self.owned else self.tensors.clone()


@dataclass(frozen=True)
class MergeMethodSpec:
    name: str
    parameters: Tuple[ParameterSpec, ...] = ()
    contract: InputContract = field(default_factory=InputContract)
    pretty_name: Optional[str] = None
    reference_url: Optional[str] = None
    optional_tensor_policy: OptionalTensorPolicy = OptionalTensorPolicy.ERROR
    uses_accelerator: bool = True

    def __post_init__(self):
        object.__setattr__(self, "parameters", tuple(self.parameters))
        names = [parameter.name for parameter in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate parameter names for merge method {self.name}")
        if "batch_options" in names:
            raise ValueError("batch_options is reserved for execution settings")
        if self.contract.base == BasePolicy.IGNORED and any(
            p.input_target == InputParameterTarget.NON_BASE for p in self.parameters
        ):
            raise ValueError("Non-base parameters require a base-aware input contract")

    @property
    def shared_parameters(self) -> Tuple[ParameterSpec, ...]:
        return tuple(p for p in self.parameters if p.scope == ParameterScope.SHARED)

    @property
    def input_parameters(self) -> Tuple[ParameterSpec, ...]:
        return tuple(p for p in self.parameters if p.scope == ParameterScope.INPUT)


class MergeMethod(ABC):
    """Logical-batch binding and validation shared by both execution strategies."""

    spec: MergeMethodSpec
    supports_batching: bool = False

    def validate_inputs(
        self,
        input_ids: Sequence[Hashable],
        base_id: Optional[Hashable] = None,
        *,
        group_name: Optional[str] = None,
    ) -> None:
        self.spec.contract.validate_ids(input_ids, base_id, group_name=group_name)

    def __call__(
        self,
        batch: MergeBatch,
        /,
        *,
        batch_options: Optional[BatchOptions] = None,
        **parameters: Any,
    ) -> MergedBatch:
        # Validate every group before running any tensor math.
        for group in batch.groups:
            self.validate_inputs(
                [entry.id for entry in group.entries],
                group.base.id if group.base else None,
                group_name=group.metadata.name,
            )
            group.validate_tensors()

        unknown = set(parameters) - {p.name for p in self.spec.parameters}
        if unknown:
            raise TypeError(
                f"Unknown parameter(s) for {self.spec.name}: {', '.join(sorted(unknown))}"
            )

        # Parameter validation is also two-phase so a malformed later group cannot
        # leave callers with a partially executed batch.
        bound_parameters = [
            self._bind_group_parameters(
                group, parameters, group_index, len(batch.groups)
            )
            for group_index, group in enumerate(batch.groups)
        ]

        return self._execute(batch, bound_parameters, batch_options or BatchOptions())

    @abstractmethod
    def _execute(
        self, batch: MergeBatch, parameters: List[Dict[str, Any]], options: BatchOptions
    ) -> MergedBatch: ...

    def _bind_group_parameters(
        self,
        group: TensorGroup,
        supplied: Mapping[str, Any],
        group_index: int,
        group_count: int,
    ) -> Dict[str, Any]:
        result = {}
        for parameter in self.spec.parameters:
            if parameter.name in supplied:
                value = supplied[parameter.name]
            elif (
                parameter.scope == ParameterScope.INPUT
                and parameter.input_target == InputParameterTarget.NON_BASE
                and not group.non_base
            ):
                value = {}
            elif parameter.required:
                raise TypeError(
                    f"Missing required parameter {parameter.name} for {self.spec.name}"
                )
            else:
                value = parameter.default

            if isinstance(value, PerGroupValues):
                if len(value.values) != group_count:
                    raise ValueError(
                        f"Parameter {parameter.name} expects {group_count} group "
                        f"values; got {len(value.values)}"
                    )
                value = value.values[group_index]

            if parameter.scope == ParameterScope.SHARED:
                result[parameter.name] = parameter.validate(value)
            else:
                result[parameter.name] = self._bind_per_input(group, parameter, value)
        return result

    def _bind_per_input(
        self, group: TensorGroup, parameter: ParameterSpec, value: Any
    ) -> PerInputValues:
        entries = (
            group.non_base
            if parameter.input_target == InputParameterTarget.NON_BASE
            else group.entries
        )
        ids = [entry.id for entry in entries]
        if isinstance(value, PerInputValues):
            raw_values = [value[key] for key in ids]
        elif isinstance(value, Mapping):
            missing = [key for key in ids if key not in value]
            if missing:
                raise ValueError(f"Missing {parameter.name} for input(s): {missing}")
            raw_values = [value[key] for key in ids]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != len(ids):
                raise ValueError(
                    f"Parameter {parameter.name} expects {len(ids)} values; "
                    f"got {len(value)}"
                )
            raw_values = list(value)
        else:
            # Scalar defaults and programmatic scalar arguments broadcast.
            raw_values = [value] * len(ids)

        return PerInputValues(
            [(key, parameter.validate(raw)) for key, raw in zip(ids, raw_values)]
        )


class BatchedMergeMethod(MergeMethod):
    """A method with a native numerical batch kernel."""

    supports_batching = True

    def __init__(
        self, spec: MergeMethodSpec, implementation: Callable[..., torch.Tensor]
    ):
        self.spec = spec
        self.implementation = implementation

    def merge_batch(self, batch: TensorBatch, **parameters: Any) -> torch.Tensor:
        """Execute already-aligned numerical arguments (no logical binding)."""
        result = self.implementation(batch, **parameters)
        expected = (batch.tensors.shape[0], *batch.tensors.shape[2:])
        if not isinstance(result, torch.Tensor) or result.shape != expected:
            raise TypeError(
                f"Merge method {self.spec.name} must return a tensor of shape {expected}"
            )
        return result

    def _execute(
        self, batch: MergeBatch, parameters: List[Dict[str, Any]], options: BatchOptions
    ) -> MergedBatch:
        from mergekit.merge_methods.batching import prepare_batches

        # Preparation validates every group before any kernel is invoked. Packing
        # happens one chunk at a time. Outputs that alias a workspace may retain
        # its storage, just like outputs with retained autograd graphs.
        prepared = prepare_batches(batch, parameters, self.spec, options)
        results = [None] * len(batch.groups)
        for chunk in prepared:
            packed, kwargs = chunk.pack(self.spec)
            merged = self.merge_batch(packed, **kwargs)
            for index, tensor in zip(chunk.indices, merged.unbind(0)):
                results[index] = tensor
            del packed, kwargs, merged
        return MergedBatch(tensors=tuple(results))


class GroupMergeMethod(MergeMethod):
    """A method that operates on one logical tensor group at a time.

    Group methods deliberately avoid packing: a sequential kernel should not pay
    the memory cost of packing or lose access to logical tensor metadata.
    """

    @abstractmethod
    def merge_group(self, group: TensorGroup, **parameters: Any) -> torch.Tensor: ...

    def _execute(
        self, batch: MergeBatch, parameters: List[Dict[str, Any]], options: BatchOptions
    ) -> MergedBatch:
        results = []
        for group, kwargs in zip(batch.groups, parameters):
            result = self.merge_group(group, **kwargs)
            expected = group.entries[0].tensor.shape if group.entries else None
            if not isinstance(result, torch.Tensor) or (
                expected is not None and result.shape != expected
            ):
                raise TypeError(
                    f"Merge method {self.spec.name} must return a tensor"
                    f" of shape {expected}"
                )
            results.append(result)
        return MergedBatch(tensors=tuple(results))


class FunctionalGroupMergeMethod(GroupMergeMethod):
    def __init__(
        self, spec: MergeMethodSpec, implementation: Callable[..., torch.Tensor]
    ):
        self.spec = spec
        self.implementation = implementation

    def merge_group(self, group: TensorGroup, **parameters: Any) -> torch.Tensor:
        return self.implementation(group, **parameters)
