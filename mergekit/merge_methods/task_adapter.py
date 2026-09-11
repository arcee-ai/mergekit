# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Computation-graph adapters for consumer-neutral merge methods."""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
from typing_extensions import TypeAlias

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods.base import (
    MergeBatch,
    OptionalTensorPolicy,
    TensorEntry,
    TensorGroup,
    TensorMetadata,
)
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
    base_model: Optional[ModelReference]
    output_weight: WeightInfo
    parameters: ImmutableMap[str, Any]
    input_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]

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
            TensorEntry(
                id=model,
                tensor=tensors[model],
                is_base=model == self.base_model,
            )
            for model in self.model_order
            if model in tensors
        )
        group = TensorGroup(
            entries=entries,
            metadata=TensorMetadata.from_weight_info(self.output_weight),
        )
        method = merge_methods.get(self.method_name)
        if self.output_weight.optional and len(entries) < len(self.model_order):
            policy = method.spec.optional_tensor_policy
            if (
                len(entries) == 1
                and policy == OptionalTensorPolicy.PASSTHROUGH_SINGLETON
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
        kwargs = dict(self.parameters.items())
        for parameter in method.spec.input_parameters:
            kwargs[parameter.name] = {
                entry.id: self.input_parameters[entry.id][parameter.name]
                for entry in entries
                if entry.id in self.input_parameters
                and parameter.name in self.input_parameters[entry.id]
            }
        return method(MergeBatch(groups=(group,)), **kwargs).one()
