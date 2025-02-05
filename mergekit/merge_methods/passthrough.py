# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import Any, Dict, List, Optional

import torch
from typing_extensions import override

from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)


class PassthroughMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        if len(tensors) != 1:
            raise RuntimeError("Passthrough merge expects exactly one tensor")

        model, tensor = list(tensors.items())[0]
        scale = self.tensor_parameters[model].data.get("scale", None)
        if scale is not None:
            tensor = tensor * scale

        return tensor

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class PassthroughMerge(MergeMethod):
    def name(self) -> str:
        return "passthrough"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Passthrough"

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="scale", required=False, default_value=None)]

    def make_task(
        self,
        *,
        tensors: MergeTensorInput,
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **kwargs,
    ) -> Task:
        return PassthroughMergeTask(
            gather_tensors=tensors, tensor_parameters=tensor_parameters
        )
