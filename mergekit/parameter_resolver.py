# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Resolve configured parameters for both graph planners.

Adapters provide tensor names and settings in descending precedence. Filtering,
gradients, defaults, validation, and per-input targeting are shared here.
"""

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from mergekit.common import ModelReference
from mergekit.config import ParameterSetting, evaluate_setting
from mergekit.merge_methods.base import (
    InputParameterTarget,
    MergeMethodSpec,
    ParameterSpec,
)

Settings = Optional[Mapping[str, ParameterSetting]]


def resolve_parameter(
    parameter: ParameterSpec,
    settings: Iterable[Settings],
    *,
    tensor_name: str,
    t: float = 0,
    model: Optional[ModelReference] = None,
) -> Any:
    for scope in settings:
        if scope is not None and parameter.name in scope:
            value = evaluate_setting(
                tensor_name, scope[parameter.name], t, validate=parameter.validate
            )
            if value is not None:
                return value
    if parameter.required:
        location = f"{model}.{tensor_name}" if model is not None else tensor_name
        raise RuntimeError(
            f"Missing required parameter {parameter.name} for {location}"
        )
    return parameter.validate(parameter.default)


def resolve_parameters(
    spec: MergeMethodSpec,
    *,
    tensor_name: str,
    inputs: Mapping[ModelReference, str],
    sources: Callable[[Optional[ModelReference]], Iterable[Settings]],
    base_model: Optional[ModelReference] = None,
    t: float = 0,
) -> Tuple[Dict[str, Any], Dict[ModelReference, Dict[str, Any]]]:
    shared_settings = tuple(sources(None))
    shared = {
        p.name: resolve_parameter(p, shared_settings, tensor_name=tensor_name, t=t)
        for p in spec.shared_parameters
    }
    per_input = {}
    for model, input_name in inputs.items():
        settings = tuple(sources(model))
        per_input[model] = {
            p.name: resolve_parameter(
                p, settings, tensor_name=input_name, model=model, t=t
            )
            for p in spec.input_parameters
            if model != base_model or p.input_target == InputParameterTarget.ALL
        }
    return shared, per_input
