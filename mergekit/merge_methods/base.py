# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Consumer-neutral interfaces for tensor merge methods.

Merge methods operate on in-memory tensors. Configuration readers, model loaders, and
the computation graph are adapters around this module rather than part of the method
interface itself.
"""

from __future__ import annotations

import inspect
from abc import ABC
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import torch
from pydantic import TypeAdapter
from typing_extensions import Annotated, TypeAlias, get_args, get_origin, get_type_hints


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

    @property
    def required(self) -> bool:
        return self.default is MISSING

    @property
    def default_value(self) -> Any:
        """Compatibility alias for the old parameter definition API."""
        return None if self.required else self.default

    def validate(self, value: Any) -> Any:
        return TypeAdapter(self.value_type).validate_python(value)


class BasePolicy(str, Enum):
    OPTIONAL = "optional"
    REQUIRED = "required"
    FORBIDDEN = "forbidden"
    IGNORED = "ignored"


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
class MergeMethodSpec:
    name: str
    parameters: Tuple[ParameterSpec, ...] = ()
    contract: InputContract = field(default_factory=InputContract)
    pretty_name: Optional[str] = None
    reference_url: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "parameters", tuple(self.parameters))
        names = [parameter.name for parameter in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate parameter names for merge method {self.name}")


class MergeMethod(ABC):
    """A callable, backend-independent merge method."""

    def name(self) -> str:
        return self.spec.name

    def pretty_name(self) -> Optional[str]:
        return self.spec.pretty_name

    def reference_url(self) -> Optional[str]:
        return self.spec.reference_url

    def parameters(self) -> List[ParameterSpec]:
        """Compatibility view of shared parameters."""
        return [p for p in self.spec.parameters if p.scope == ParameterScope.SHARED]

    def tensor_parameters(self) -> List[ParameterSpec]:
        """Compatibility view of per-input parameters."""
        return [p for p in self.spec.parameters if p.scope == ParameterScope.INPUT]

    def validate_inputs(
        self,
        input_ids: Sequence[Hashable],
        base_id: Optional[Hashable] = None,
        *,
        group_name: Optional[str] = None,
    ) -> None:
        self.spec.contract.validate_ids(input_ids, base_id, group_name=group_name)

    def __call__(self, batch: MergeBatch, /, **parameters: Any) -> MergedBatch:
        # Validate every group before running any tensor math.
        for group in batch.groups:
            self.validate_inputs(
                [entry.id for entry in group.entries],
                group.base.id if group.base else None,
                group_name=group.metadata.name,
            )

        unknown = set(parameters) - {p.name for p in self.spec.parameters}
        if unknown:
            raise TypeError(
                f"Unknown parameter(s) for {self.name()}: {', '.join(sorted(unknown))}"
            )

        # Parameter validation is also two-phase so a malformed later group cannot
        # leave callers with a partially executed batch.
        bound_parameters = [
            self._bind_group_parameters(
                group, parameters, group_index, len(batch.groups)
            )
            for group_index, group in enumerate(batch.groups)
        ]

        results = []
        for group, kwargs in zip(batch.groups, bound_parameters):
            result = self.merge(group, **kwargs)
            if not isinstance(result, torch.Tensor):
                raise TypeError(
                    f"Merge method {self.name()} returned {type(result).__name__}, "
                    "expected torch.Tensor"
                )
            results.append(result)
        return MergedBatch(tensors=tuple(results))

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
            elif parameter.required:
                raise TypeError(
                    f"Missing required parameter {parameter.name} for {self.name()}"
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

    def merge(self, group: TensorGroup, **parameters: Any) -> torch.Tensor:
        raise NotImplementedError


class FunctionalMergeMethod(MergeMethod):
    def __init__(
        self, spec: MergeMethodSpec, implementation: Callable[..., torch.Tensor]
    ):
        self.spec = spec
        self.implementation = implementation

    def merge(self, group: TensorGroup, **parameters: Any) -> torch.Tensor:
        return self.implementation(group, **parameters)


def method_from_function(
    func: Callable[..., torch.Tensor],
    *,
    name: str,
    pretty_name: Optional[str] = None,
    reference_url: Optional[str] = None,
    contract: Optional[InputContract] = None,
) -> FunctionalMergeMethod:
    """Build a merge method whose parameter schema is derived from its signature."""

    signature = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    positional = list(signature.parameters.values())
    if not positional:
        raise TypeError("Merge method implementation must accept a TensorGroup")
    group_parameter = positional[0]
    if hints.get(group_parameter.name) is not TensorGroup:
        raise TypeError("First merge method argument must be annotated TensorGroup")

    specs = []
    for argument in positional[1:]:
        annotation = hints.get(argument.name)
        if annotation is None:
            raise TypeError(f"Parameter {argument.name} must have a type annotation")
        if get_origin(annotation) is not Annotated:
            raise TypeError(
                f"Parameter {argument.name} must use Shared[T], PerInput[T], "
                "or PerNonBase[T]"
            )
        annotated_args = get_args(annotation)
        markers = [m for m in annotated_args[1:] if isinstance(m, ParameterMarker)]
        if len(markers) != 1:
            raise TypeError(
                f"Parameter {argument.name} must have exactly one parameter scope"
            )
        value_type = annotated_args[0]
        marker = markers[0]
        if marker.scope == ParameterScope.INPUT:
            if get_origin(value_type) is not PerInputValues:
                raise TypeError(f"PerInput parameter {argument.name} has invalid type")
            value_type = get_args(value_type)[0]
        default = (
            MISSING if argument.default is inspect.Parameter.empty else argument.default
        )
        specs.append(
            ParameterSpec(
                name=argument.name,
                value_type=value_type,
                scope=marker.scope,
                input_target=marker.target,
                default=default,
            )
        )

    if hints.get("return") is not torch.Tensor:
        raise TypeError("Merge method return type must be torch.Tensor")

    return FunctionalMergeMethod(
        spec=MergeMethodSpec(
            name=name,
            pretty_name=pretty_name,
            reference_url=reference_url,
            contract=contract or InputContract(),
            parameters=tuple(specs),
        ),
        implementation=func,
    )


@dataclass(frozen=True)
class ConfigParameterDef:
    """Legacy parameter declaration retained while class-based methods migrate."""

    name: str
    required: bool = False
    default_value: Any = None
