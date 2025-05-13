# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import inspect
import typing
from typing import Any, Dict, List, Optional

import pydantic
import torch
from pydantic import Field
from typing_extensions import Callable

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.registry import REGISTERED_MERGE_METHODS

STANDARD_KWARGS = {"output_weight", "base_model"}


def __merge_method(
    func: Callable,
    name: str,
    reference_url: Optional[str] = None,
    pretty_name: Optional[str] = None,
) -> Callable:
    use_base_tensor_arg = False
    require_base_tensor = False
    used_kwargs = set()
    parameters: List[ConfigParameterDef] = []
    tensor_parameters: List[ConfigParameterDef] = []
    sig = inspect.signature(func)
    if "tensors" not in sig.parameters:
        raise ValueError("Merge methods must have a 'tensors' parameter")
    tensor_param = sig.parameters["tensors"]
    if (
        (tensor_param.annotation is None)
        or (not hasattr(tensor_param.annotation, "__origin__"))
        or not (
            tensor_param.annotation.__origin__ == list
            and tensor_param.annotation.__args__ == (torch.Tensor,)
        )
    ):
        raise ValueError("'tensors' must be annotated with List[torch.Tensor]")
    if "base_tensor" in sig.parameters:
        bt_param = sig.parameters["base_tensor"]
        if bt_param.annotation == torch.Tensor:
            require_base_tensor = True
        elif (
            hasattr(bt_param.annotation, "__origin__")
            and bt_param.annotation.__origin__ == typing.Union
            and (
                bt_param.annotation.__args__ == (torch.Tensor, type(None))
                or bt_param.annotation.__args__ == (type(None), torch.Tensor)
            )
        ):
            require_base_tensor = False
        else:
            raise ValueError(
                "'base_tensor' must be annotated either torch.Tensor or Optional[torch.Tensor]"
            )
        use_base_tensor_arg = True
    for arg, arg_info in sig.parameters.items():
        if arg in ("base_tensor", "tensors"):
            continue
        if arg in STANDARD_KWARGS:
            used_kwargs.add(arg)
        else:
            if arg_info.annotation is None:
                raise ValueError(
                    "All merge method arguments must have type annotations"
                )
            elif arg_info.annotation in (int, float, bool):
                default_value = arg_info.default
                if default_value == inspect.Parameter.empty:
                    default_value = None
                    required = True
                else:
                    required = False
                parameters.append(
                    ConfigParameterDef(
                        name=arg, required=required, default_value=default_value
                    )
                )
            elif (
                hasattr(arg_info.annotation, "__origin__")
                and arg_info.annotation.__origin__ == list
                and arg_info.annotation.__args__[0] in (float, int)
            ):
                default_value = arg_info.default
                if default_value == inspect.Parameter.empty:
                    default_value = None
                    required = True
                else:
                    required = False
                if (not required) and (not isinstance(default_value, (int, float))):
                    raise ValueError(
                        f"Unexpected default for presumed tensor parameter '{arg}' - should be single number, got {repr(default_value)}"
                    )
                tensor_parameters.append(
                    ConfigParameterDef(
                        name=arg, required=required, default_value=default_value
                    )
                )

    tt_fields = {}
    tt_fields["gather_tensors"] = (MergeTensorInput, Field(...))
    if ("base_model" in used_kwargs) or use_base_tensor_arg:
        bm_ty = ModelReference if require_base_tensor else Optional[ModelReference]
        field_kwargs = {"default": None} if not require_base_tensor else {}
        tt_fields["base_model"] = (bm_ty, Field(**field_kwargs))
    if "output_weight" in used_kwargs:
        tt_fields["output_weight"] = (WeightInfo, Field(...))
    if parameters:
        tt_fields["parameters"] = (ImmutableMap[str, Any], Field(...))
    if tensor_parameters:
        tt_fields["tensor_parameters"] = (
            ImmutableMap[ModelReference, ImmutableMap[str, Any]],
            Field(...),
        )

    def _arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    tt_fields["arguments"] = _arguments

    def _group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()

    tt_fields["group_label"] = _group_label

    def _uses_accelerator(self) -> bool:
        return True

    tt_fields["uses_accelerator"] = _uses_accelerator

    def _execute(self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs):
        model_refs = set(tensors.keys())
        base_model = getattr(self, "base_model", None)
        if base_model and base_model in model_refs:
            model_refs.remove(base_model)
            if not use_base_tensor_arg:
                model_refs = [base_model] + list(model_refs)
            else:
                model_refs = list(model_refs)
        base_tensor = tensors.get(base_model, None)
        tensors = [tensors[key] for key in model_refs]
        inner_kwargs = {}
        for key in used_kwargs:
            inner_kwargs[key] = getattr(self, key)
        if use_base_tensor_arg:
            inner_kwargs["base_tensor"] = base_tensor
            if require_base_tensor and (inner_kwargs["base_tensor"] is None):
                raise ValueError("Base model tensor required but not present")
        for key in parameters:
            inner_kwargs[key.name] = self.parameters[key.name]
        for key in tensor_parameters:
            inner_kwargs[key.name] = [
                self.tensor_parameters[ref][key.name] for ref in model_refs
            ]
        return func(tensors=tensors, **inner_kwargs)

    tt_fields["execute"] = _execute

    tt_name = f"{name.title().replace(' ', '')}MergeTask"
    tt_cls = pydantic.create_model(tt_name, __base__=Task[torch.Tensor], **tt_fields)

    mm_fields = {}

    def _make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
        **_kwargs,
    ) -> Task:
        tt_kwargs = {"gather_tensors": tensors}
        if "base_model" in tt_fields:
            tt_kwargs["base_model"] = base_model
        if "output_weight" in tt_fields:
            tt_kwargs["output_weight"] = output_weight
        if "parameters" in tt_fields:
            tt_kwargs["parameters"] = parameters
        if "tensor_parameters" in tt_fields:
            tt_kwargs["tensor_parameters"] = tensor_parameters
        return tt_cls(**tt_kwargs)

    mm_fields["make_task"] = _make_task

    def _name(self) -> str:
        return name

    mm_fields["name"] = _name

    def _pretty_name(self) -> Optional[str]:
        return pretty_name

    mm_fields["pretty_name"] = _pretty_name

    def _reference_url(self) -> Optional[str]:
        return reference_url

    mm_fields["reference_url"] = _reference_url

    def _tensor_parameters(self) -> List[ConfigParameterDef]:
        return tensor_parameters

    mm_fields["tensor_parameters"] = _tensor_parameters

    def _parameters(self) -> List[ConfigParameterDef]:
        return parameters

    mm_fields["parameters"] = _parameters

    mm_name = f"{name.title().replace(' ', '')}MergeMethod"
    mm_cls = type(mm_name, (MergeMethod,), mm_fields)
    REGISTERED_MERGE_METHODS[name] = mm_cls()
    return func


def merge_method(
    name: str,
    reference_url: Optional[str] = None,
    pretty_name: Optional[str] = None,
) -> Callable:
    """Decorator for registering custom model merging algorithms.

    Enables creation of new merge algorithms that can be specified in merge configurations
    and executed through mergekit's processing pipeline. Handles parameter validation, task
    creation, and registration in the mergekit system.

    Args:
        name: Unique identifier for the merge method (lowercase, snake_case recommended)
        reference_url: Optional URL to paper/documentation explaining the method (used in generated READMEs)
        pretty_name: Human-readable display name (used in generated READMEs)

    Returns:
        A decorator that registers the function as a merge method implementation

    Notes:
        The decorated function must meet these requirements:
        - First parameter must be `tensors: List[torch.Tensor]`
        - Must return a single `torch.Tensor`
        - All parameters must have type annotations

        Key behavioral considerations:

        *Base Model Handling:*
        - If the method includes a `base_tensor` parameter:
            * `torch.Tensor` annotation: Requires `base_model` in config, receives its tensor
            * `Optional[torch.Tensor]` annotation: `base_model` optional, `None` if not provided
            * Non-base model tensors passed in `tensors` list
        - Without `base_tensor` parameter:
            * Base model tensor (if specified) will be first in `tensors` list

        *Parameter Types:*
        - Standard parameters (auto-populated):
            * `base_tensor`: Tensor from base model (type determines requirement)
            * `output_weight`: WeightInfo with output configuration
            * `base_model`: ModelReference if using base model logic
        - Scalar parameters (global config):
            * `float`, `int`, or `bool` types specified in top-level `parameters`
        - Tensor parameters (per-model weights):
            * Annotated as `List[float]` or `List[int]`
            * Configured per-model in their `parameters` section
            * Collected into lists ordered by input models

    Example:
        ```python
        @merge_method(
            name="average",
            pretty_name="Simple Average",
            reference_url="https://example.com/mean-merge"
        )
        def average_merge(
            tensors: List[torch.Tensor],  # Input tensors to merge
            weights: List[float],         # Per-model weights (tensor parameter)
            normalize: bool = True        # Scalar parameter
        ) -> torch.Tensor:
            if normalize:
                weights = [w / sum(weights) for w in weights]
            return sum(t * w for t, w in zip(tensors, weights))
        ```

        This would enable merge configurations like:
        ```yaml
        merge_method: average
        models:
          - model: model_a
            parameters:
              weights: 0.3
          - model: model_b
            parameters:
              weights: [0.6, 0.8]
        parameters:
          normalize: true
        ```

    Raises:
        ValueError: If function signature doesn't meet requirements
        TypeError: For invalid parameter annotations
    """

    def _wrap(func: Callable) -> Callable:
        return __merge_method(func, name, reference_url, pretty_name)

    return _wrap
