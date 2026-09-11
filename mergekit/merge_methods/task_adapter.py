# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Computation-graph adapters for merge methods."""

import logging
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from typing_extensions import TypeAlias

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference, dtype_from_name
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods.base import (
    OptionalTensorPolicy,
    PerInputValues,
    TensorEntry,
    TensorGroup,
    TensorMetadata,
)
from mergekit.merge_methods.buffers import copy_non_floating_buffer
from mergekit.tokenizer import PermutedEmbeddings


class TensorDictWrapper(Task[Dict[ModelReference, torch.Tensor]]):
    """Graph adapter used by the raw-PyTorch merge command."""

    tensors: ImmutableMap[ModelReference, Task[torch.Tensor]]

    def arguments(self) -> Dict[str, Task]:
        return {
            key.model_dump_json(
                exclude_none=True, exclude_defaults=True, round_trip=True
            ): value
            for key, value in self.tensors.items()
        }

    def execute(self, **kwargs) -> Dict[ModelReference, torch.Tensor]:
        return {
            ModelReference.model_validate_json(key): value
            for key, value in kwargs.items()
        }


MergeTensorInput: TypeAlias = Union[
    GatherTensors, PermutedEmbeddings, TensorDictWrapper
]


class ExecuteMergeMethodTask(Task[Optional[torch.Tensor]]):
    """Adapt graph dependencies to the callable merge method interface."""

    method_name: str
    gather_tensors: MergeTensorInput
    model_order: Tuple[ModelReference, ...]
    base_index: Optional[int]
    output_weight: WeightInfo
    parameters: ImmutableMap[str, Any]
    dtype: Optional[str] = None
    out_dtype: Optional[str] = None

    @classmethod
    def from_parameters(
        cls,
        *,
        model_order: Tuple[ModelReference, ...],
        base_model: Optional[ModelReference],
        parameters: Mapping[str, Any],
        input_parameters: Mapping[ModelReference, Mapping[str, Any]],
        **kwargs,
    ) -> "ExecuteMergeMethodTask":
        """Bind already-resolved planner parameters to stable input positions.

        Retaining these positions when optional inputs are absent keeps
        coefficients attached to the right tensors.
        """
        from mergekit import merge_methods

        method = merge_methods.get(kwargs["method_name"])
        method.validate_inputs(
            model_order, base_model, group_name=kwargs["output_weight"].name
        )
        bound = dict(parameters.items())
        for parameter in method.spec.input_parameters:
            bound[parameter.name] = PerInputValues(
                [
                    (index, input_parameters[model][parameter.name])
                    for index, model in enumerate(model_order)
                    if model in input_parameters
                    and parameter.name in input_parameters[model]
                ]
            )
        return cls(
            model_order=model_order,
            base_index=(
                model_order.index(base_model)
                if base_model is not None and base_model in model_order
                else None
            ),
            parameters=ImmutableMap(bound),
            **kwargs,
        )

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()

    def uses_accelerator(self) -> bool:
        from mergekit import merge_methods

        return merge_methods.get(self.method_name).spec.uses_accelerator

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> Optional[torch.Tensor]:
        from mergekit import merge_methods

        entries = tuple(
            TensorEntry(id=index, tensor=tensor, is_base=index == self.base_index)
            for index, model in enumerate(self.model_order)
            if (tensor := tensors.get(model)) is not None
        )
        group = TensorGroup(
            entries=entries,
            metadata=TensorMetadata.from_weight_info(self.output_weight),
        )
        method = merge_methods.get(self.method_name)
        if self.output_weight.optional and len(entries) < len(self.model_order):
            policy = method.spec.optional_tensor_policy
            if len(entries) == 1 and (
                policy == OptionalTensorPolicy.PASSTHROUGH_SINGLETON
                or (
                    policy == OptionalTensorPolicy.PASSTHROUGH_BASE_SINGLETON
                    and entries[0].is_base
                )
            ):
                return entries[0].tensor
            if (
                policy == OptionalTensorPolicy.BASE_OR_SKIP
                and len(entries) < method.spec.contract.min_inputs
            ):
                if len(entries) == 1 and entries[0].is_base:
                    return entries[0].tensor
                logging.warning(
                    "Skipping optional weight %s: insufficient inputs",
                    self.output_weight.name,
                )
                return None
        # A missing configured base is not the same as choosing a baseless
        # algorithm. TensorGroup alone cannot retain that distinction once the
        # loader has omitted the base's optional weight.
        method.validate_inputs(
            [entry.id for entry in entries],
            self.base_index,
            group_name=self.output_weight.name,
        )
        buffer = copy_non_floating_buffer(group.tensors, self.output_weight.name)
        if buffer is not None:
            return buffer
        group.validate_tensors()
        parameters = dict(self.parameters.items())
        if len(entries) < len(self.model_order):
            parameters = {
                name: (
                    PerInputValues(
                        [
                            (entry.id, value[entry.id])
                            for entry in entries
                            if entry.id in value
                        ]
                    )
                    if isinstance(value, PerInputValues)
                    else value
                )
                for name, value in parameters.items()
            }
        (result,) = method._execute_resolved(
            (group,),
            [parameters],
            dtype=dtype_from_name(self.dtype),
            out_dtype=dtype_from_name(self.out_dtype),
        )
        return result
