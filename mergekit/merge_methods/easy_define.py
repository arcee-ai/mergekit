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
    OptionMarker,
    ParameterMarker,
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
    if get_origin(annotation) is not Annotated:
        raise TypeError(f"Parameter {argument.name} must have a scope annotation")
    value_type, *metadata = get_args(annotation)
    markers = [
        m
        for m in metadata
        if isinstance(m, (ParameterMarker, BatchParameter, OptionMarker))
    ]
    if len(markers) != 1:
        raise TypeError(
            f"Parameter {argument.name} must have exactly one scope annotation"
        )
    marker = markers[0]
    remaining = [m for m in metadata if m is not marker]
    scope = ParameterScope.SHARED
    kwargs = {}
    if batched:
        if isinstance(marker, BatchParameter):
            if value_type is not torch.Tensor or remaining:
                raise TypeError(
                    "BatchParameter annotates torch.Tensor; put value constraints inside BatchParameter(value_type)"
                )
            value_type = marker.value_type
            scalar_type = value_type
            while get_origin(scalar_type) is Annotated:
                scalar_type = get_args(scalar_type)[0]
            if scalar_type not in (float, int, bool):
                raise TypeError(
                    "BatchParameter requires a bool, int, or float scalar type"
                )
            scope = marker.scope
            kwargs = {"input_target": marker.target, "batch_tensor": True}
        elif not isinstance(marker, OptionMarker):
            raise TypeError(
                "Batch kernels use BatchParameter or Option[T], not group-kernel annotations"
            )
    else:
        if not isinstance(marker, ParameterMarker):
            raise TypeError(
                "Group kernels use Shared[T], PerInput[T], or PerNonBase[T]"
            )
        scope = marker.scope
        kwargs = {"input_target": marker.target}
        if scope == ParameterScope.INPUT:
            if get_origin(value_type) is not PerInputValues or remaining:
                raise TypeError("Put per-input value constraints inside PerInput[T]")
            value_type = get_args(value_type)[0]
    if remaining:
        value_type = Annotated[(value_type, *remaining)]
    return ParameterSpec(
        name=argument.name,
        value_type=value_type,
        scope=scope,
        default=(
            MISSING if argument.default is inspect.Parameter.empty else argument.default
        ),
        **kwargs,
    )


def _spec_from_function(
    func: Callable[..., torch.Tensor],
    *,
    batched: bool,
    name: str,
    pretty_name: Optional[str] = None,
    reference_url: Optional[str] = None,
    contract: Optional[InputContract] = None,
    optional_tensor_policy: OptionalTensorPolicy = OptionalTensorPolicy.ERROR,
    uses_accelerator: bool = True,
) -> MergeMethodSpec:
    arguments = list(inspect.signature(func).parameters.values())
    hints = get_type_hints(func, include_extras=True)
    input_type = TensorBatch if batched else TensorGroup
    if not arguments or hints.get(arguments[0].name) is not input_type:
        raise TypeError(
            f"First kernel argument must be annotated {input_type.__name__}"
        )
    if arguments[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError("First kernel argument must accept a positional value")
    if hints.get("return") is not torch.Tensor:
        raise TypeError("Merge kernel return type must be torch.Tensor")
    return MergeMethodSpec(
        name=name,
        pretty_name=pretty_name,
        reference_url=reference_url,
        contract=contract or InputContract(),
        optional_tensor_policy=optional_tensor_policy,
        uses_accelerator=uses_accelerator,
        parameters=tuple(
            _parameter_spec(arg, hints.get(arg.name), batched) for arg in arguments[1:]
        ),
    )


def from_batch_kernel(
    func: Callable[..., torch.Tensor], **spec_options: Any
) -> BatchedMergeMethod:
    """Build a method whose primitive operation is a numerical TensorBatch."""
    return BatchedMergeMethod(
        _spec_from_function(func, batched=True, **spec_options), func
    )


def from_group_kernel(
    func: Callable[..., torch.Tensor], **spec_options: Any
) -> FunctionalGroupMergeMethod:
    """Explicitly lift a sequential group kernel over a logical batch."""
    return FunctionalGroupMergeMethod(
        _spec_from_function(func, batched=False, **spec_options), func
    )


def _register(factory: Callable[..., MergeMethod], **spec_options: Any):
    def wrap(func: Callable[..., torch.Tensor]) -> MergeMethod:
        # Registry construction imports built-ins, so defer the registry import.
        from mergekit.merge_methods.registry import REGISTERED_MERGE_METHODS

        method = factory(func, **spec_options)
        if method.spec.name in REGISTERED_MERGE_METHODS:
            raise ValueError(f"Merge method {method.spec.name!r} is already registered")
        REGISTERED_MERGE_METHODS[method.spec.name] = method
        return method

    return wrap


def merge_method(**spec_options: Any):
    """Register a native batch kernel."""
    return _register(from_batch_kernel, **spec_options)


def group_merge_method(**spec_options: Any):
    """Register a method that operates on one logical tensor group at a time."""
    return _register(from_group_kernel, **spec_options)
