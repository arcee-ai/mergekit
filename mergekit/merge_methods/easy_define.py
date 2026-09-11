# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Signature-derived batch kernels and explicitly sequential group adapters."""

import inspect
from typing import Any, Callable, Optional

import torch
from typing_extensions import Annotated, get_args, get_origin, get_type_hints

from mergekit.merge_methods.base import (
    MISSING,
    BatchedMergeMethod,
    BatchParameter,
    FunctionalGroupMergeMethod,
    InputContract,
    MergeMethod,
    MergeMethodSpec,
    OptionalTensorPolicy,
    ParameterScope,
    ParameterSpec,
    PerInputValues,
    TensorBatch,
    TensorGroup,
)


def _parameter_spec(
    argument: inspect.Parameter, annotation: Any, batched: bool
) -> ParameterSpec:
    if argument.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        raise TypeError(f"Parameter {argument.name} must accept keyword arguments")
    if annotation is inspect.Parameter.empty:
        raise TypeError(f"Parameter {argument.name} must have a type annotation")
    value_type, metadata = annotation, []
    if get_origin(annotation) is Annotated:
        value_type, *metadata = get_args(annotation)
    markers = [m for m in metadata if isinstance(m, (ParameterScope, BatchParameter))]
    if len(markers) > 1:
        raise TypeError(
            f"Parameter {argument.name} must have at most one scope annotation"
        )
    marker = markers[0] if markers else None
    remaining = [m for m in metadata if m is not marker]
    scope = ParameterScope.SHARED
    if isinstance(marker, BatchParameter):
        if not batched:
            raise TypeError(
                "Group kernels use PerInput[T] or PerNonBase[T], not BatchParameter"
            )
        if value_type is not torch.Tensor or remaining:
            raise TypeError(
                "BatchParameter annotates torch.Tensor; put value constraints inside BatchParameter(value_type)"
            )
        value_type = marker.value_type
        scalar_type = value_type
        while get_origin(scalar_type) is Annotated:
            scalar_type = get_args(scalar_type)[0]
        if scalar_type not in (float, int, bool):
            raise TypeError("BatchParameter requires a bool, int, or float scalar type")
        scope = marker.scope
    elif isinstance(marker, ParameterScope):
        if batched:
            raise TypeError(
                "Batch kernels use BatchParameter, not PerInput[T] or PerNonBase[T]"
            )
        if (
            get_origin(value_type) is not PerInputValues
            or remaining
            or marker == ParameterScope.SHARED
        ):
            raise TypeError("Put per-input value constraints inside PerInput[T]")
        value_type = get_args(value_type)[0]
        scope = marker
    else:
        if value_type is torch.Tensor:
            raise TypeError("Tensor coefficients must be annotated with BatchParameter")
        if value_type is PerInputValues or get_origin(value_type) is PerInputValues:
            raise TypeError("Per-input values must use PerInput[T] or PerNonBase[T]")
        # Preserve ordinary Annotated metadata, including Pydantic constraints.
        value_type = annotation
    return ParameterSpec(
        name=argument.name,
        value_type=value_type,
        scope=scope,
        default=(
            MISSING if argument.default is inspect.Parameter.empty else argument.default
        ),
        batch_tensor=isinstance(marker, BatchParameter),
    )


def _method_from_function(
    func: Callable[..., torch.Tensor],
    *,
    name: str,
    pretty_name: Optional[str] = None,
    reference_url: Optional[str] = None,
    contract: Optional[InputContract] = None,
    optional_tensor_policy: OptionalTensorPolicy = OptionalTensorPolicy.ERROR,
    uses_accelerator: bool = True,
) -> MergeMethod:
    arguments = list(inspect.signature(func).parameters.values())
    hints = get_type_hints(func, include_extras=True)
    input_type = hints.get(arguments[0].name) if arguments else None
    if input_type not in (TensorBatch, TensorGroup):
        raise TypeError(
            "First kernel argument must be annotated TensorBatch or TensorGroup"
        )
    batched = input_type is TensorBatch
    if arguments[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError("First kernel argument must accept a positional value")
    if hints.get("return") is not torch.Tensor:
        raise TypeError("Merge kernel return type must be torch.Tensor")
    spec = MergeMethodSpec(
        name=name,
        pretty_name=pretty_name,
        reference_url=reference_url,
        contract=contract or InputContract(),
        optional_tensor_policy=optional_tensor_policy,
        uses_accelerator=uses_accelerator,
        parameters=tuple(
            _parameter_spec(arg, hints.get(arg.name, inspect.Parameter.empty), batched)
            for arg in arguments[1:]
        ),
    )

    wrapper = BatchedMergeMethod if batched else FunctionalGroupMergeMethod
    return wrapper(spec, func)


def merge_method(
    func: Optional[Callable[..., torch.Tensor]] = None, **spec_options: Any
):
    """Construct a method, directly or as a decorator, without registering it.

    The first argument's TensorBatch or TensorGroup annotation selects execution.
    Ordinary parameter annotations describe shared Python values; per-input values
    and numerical coefficient tensors require their respective scope annotations.
    Register the returned method explicitly when it needs lookup by name.
    """
    if func is None:
        return lambda implementation: _method_from_function(
            implementation, **spec_options
        )
    return _method_from_function(func, **spec_options)
