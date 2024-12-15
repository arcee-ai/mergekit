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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel
from typing_extensions import TypeAlias

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.tokenizer import PermutedEmbeddings


class TensorDictWrapper(Task[Dict[ModelReference, torch.Tensor]]):
    tensors: ImmutableMap[ModelReference, Task[torch.Tensor]]

    def arguments(self) -> Dict[str, Task]:
        return {
            k.model_dump_json(
                exclude_none=True, exclude_defaults=True, round_trip=True
            ): v
            for k, v in self.tensors.items()
        }

    def execute(self, **kwargs) -> Dict[ModelReference, torch.Tensor]:
        return {ModelReference.model_validate_json(k): v for k, v in kwargs.items()}


MergeTensorInput: TypeAlias = Union[
    GatherTensors, PermutedEmbeddings, TensorDictWrapper
]


class ConfigParameterDef(BaseModel):
    name: str
    required: bool = False
    default_value: Any = None


class MergeMethod(ABC):
    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return []

    def parameters(self) -> List[ConfigParameterDef]:
        return []

    @abstractmethod
    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
    ) -> Task:
        ...
