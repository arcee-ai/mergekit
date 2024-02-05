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
from typing import Any, List, Optional

from pydantic import BaseModel

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors


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
        tensors: GatherTensors,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
    ) -> Task:
        ...
