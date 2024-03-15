# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

from typing import Any, Dict, List

import torch
from torch._tensor import Tensor

from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods.base import ConfigParameterDef, MergeMethod


class PassthroughMergeTask(Task[torch.Tensor]):
    gather_tensors: GatherTensors
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> Tensor:
        if len(tensors) != 1:
            raise RuntimeError("Passthrough merge expects exactly one tensor")

        model, tensor = list(tensors.items())[0]
        scale = self.tensor_parameters[model].data.get("scale", None)
        if scale is not None:
            tensor = tensor * scale

        return tensor


class PassthroughMerge(MergeMethod):
    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="scale", required=False, default_value=None)]

    def make_task(
        self,
        *,
        tensors: GatherTensors,
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **kwargs,
    ) -> Task:
        return PassthroughMergeTask(
            gather_tensors=tensors, tensor_parameters=tensor_parameters
        )
