# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel
from typing_extensions import TypeAlias

from llmtailor.architecture import WeightInfo
from llmtailor.common import ImmutableMap, ModelReference
from llmtailor.graph import Task
from llmtailor.io.tasks import GatherTensors
from llmtailor.tokenizer import PermutedEmbeddings


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
    def name(self) -> str: ...

    def pretty_name(self) -> Optional[str]:
        return None

    def reference_url(self) -> Optional[str]:
        return None

    @abstractmethod
    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
    ) -> Task: ...
