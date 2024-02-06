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

from typing import Dict, Generic, List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch._tensor import Tensor
from typing_extensions import TypeVar

from mergekit.architecture import ProceduralSpaceInfo, WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


class CollectTupleTask(Task[Tuple]):
    """Collects the results of a number of tasks into a tuple."""

    tasks: Tuple[Task]

    def arguments(self) -> Dict[str, Task]:
        return {f"a_{idx}": task for idx, task in enumerate(self.tasks)}

    def execute(self, **kwargs) -> List:
        return tuple([kwargs[f"a_{idx}"] for idx in range(len(self.tasks))])

    def group_label(self) -> Optional[str]:
        return max(t.group_label() or "" for t in self.arguments().values())


class CollectDictTask(Task[Dict[KeyT, ValueT]], Generic[KeyT, ValueT]):
    """Collects a dictionary of tasks into a dictionary of results."""

    tasks: ImmutableMap[KeyT, Task[ValueT]]

    @property
    def _ordered_keys(self):
        return list(sorted(self.tasks.keys(), key=repr))

    def arguments(self) -> Dict[str, Task]:
        return {
            "elements": CollectTupleTask(
                tasks=[self.tasks[key] for key in self._ordered_keys]
            )
        }

    def execute(self, elements: List[ValueT]) -> Dict[KeyT, ValueT]:
        return dict(zip(self._ordered_keys, elements))

    def group_label(self) -> Optional[str]:
        return max(t.group_label() or "" for t in self.arguments().values())


class AlignModelToSpaceTask(Task[torch.Tensor]):
    tensors_base: Tuple[Task[torch.Tensor]]
    tensors_model: Tuple[Task[torch.Tensor]]
    input_transforms: Optional[Dict[ModelReference, Task[torch.Tensor]]] = None

    def arguments(self) -> Dict[str, Task]:
        res = {}
        if self.input_transforms:
            res["input_transforms"] = CollectDictTask(tasks=self.input_transforms)
        res["base_tensors"] = CollectTupleTask(tasks=self.tensors_base)
        res["model_tensors"] = CollectTupleTask(tasks=self.tensors_model)
        return res

    def execute(
        self,
        base_tensors: List[torch.Tensor],
        model_tensors: List[torch.Tensor],
        input_transforms: Optional[List[torch.Tensor]] = None,
    ) -> Tensor:
        out_dim = base_tensors[0].shape[0]
        if not all(t.shape[0] == out_dim for t in (base_tensors + model_tensors)):
            raise RuntimeError(
                "All tensors that share an output space must have same output dimension"
            )

        if input_transforms:
            # Apply input transformations to model tensors
            new_model_tensors = []
            for x_model, tf_in in zip(model_tensors, input_transforms):
                if tf_in:
                    new_model_tensors.append(x_model @ tf_in)
                else:
                    new_model_tensors.append(x_model)
            model_tensors = new_model_tensors

        # Solve LAP to find best permutation of model weights to base weights
        cost_mat = torch.zeros(
            out_dim, out_dim, device=base_tensors[0].device, dtype=base_tensors[0].dtype
        )
        for x_base, x_model in zip(base_tensors, model_tensors):
            cost_mat += x_base @ x_model.T

        ri, ci = linear_sum_assignment(cost_mat.numpy(), maximize=True)
        model_to_base = torch.zeros_like(cost_mat, dtype=bool)
        model_to_base[(ri, ci)] = 1
        return model_to_base


class TransposeTensor(Task[torch.Tensor]):
    tensor_task: Task[torch.Tensor]

    def arguments(self) -> Dict[str, Task]:
        return {
            "tensor": self.tensor_task,
        }

    def execute(
        self,
        tensor: torch.Tensor,
    ) -> Tensor:
        return tensor.T


class GetAlignedTensor(Task[torch.Tensor]):
    tensor_task: Task[torch.Tensor]
    transform_in: Optional[Task[torch.Tensor]] = None
    transform_out: Optional[Task[torch.Tensor]] = None

    def arguments(self) -> Dict[str, Task]:
        return {
            "tensor": self.tensor_task,
            "transform_in": self.transform_in,
            "transform_out": self.transform_out,
        }

    def execute(
        self,
        tensor: torch.Tensor,
        transform_in: Optional[torch.Tensor] = None,
        transform_out: Optional[torch.Tensor] = None,
    ) -> Tensor:
        if transform_in:
            tensor = tensor @ transform_in
        if transform_out:
            tensor = transform_out @ tensor
        return tensor


class ResidualSpaceTransform(Task[torch.Tensor]):
    input_transform_tasks: Tuple[Task[torch.Tensor]]

    def arguments(self) -> Dict[str, Task]:
        return {"transforms": CollectTupleTask(tasks=self.input_transform_tasks)}

    def execute(self, transforms: List[torch.Tensor]) -> Tensor:
        return sum(transforms) / max(len(transforms), 1)


class SpacePlanner:
    base_model: ModelReference
    space_weights: Dict[str, Dict[ModelReference, List[WeightInfo]]]
    procedural_spaces: Dict[str, ProceduralSpaceInfo]

    def __init__(self, base_model: ModelReference):
        self.base_model = base_model
        self.space_weights = {}
        self.procedural_spaces = {}

    def add_weight(
        self, weight: WeightInfo, input_weights: List[Tuple[ModelReference, WeightInfo]]
    ):
        if weight.output_space not in self.space_weights:
            self.space_weights[weight.output_space] = {}

        st = self.space_weights[weight.output_space]
        for model, input_weight in input_weights:
            if model not in self.space_weights[weight.output_space]:
                st[model] = []
            st[model].append(input_weight)

    def add_procedural_space(self, info: ProceduralSpaceInfo):
        self.procedural_spaces[info.name] = info


class DelayedAlignTask(Task[Optional[torch.Tensor]], arbitrary_types_allowed=True):
    planner: SpacePlanner
    space: str
    for_model: ModelReference

    def arguments(self) -> Dict[str, Task]:
        if self.space in self.planner.procedural_spaces:
            return {
                "transform": ResidualSpaceTransform(
                    input_transform_tasks=[
                        DelayedAlignTask(
                            planner=self.planner,
                            space=in_space,
                            for_model=self.for_model,
                        )
                        for in_space in self.planner.procedural_spaces[
                            self.space
                        ].inputs
                    ]
                )
            }

        if self.space not in self.planner.space_weights:
            return {}

        weights = self.planner.space_weights[self.space]
        model_weights = weights[self.for_model]
        base_weights = weights[self.planner.base_model]
        input_transforms = tuple(
            [
                DelayedAlignTask(
                    planner=self.planner,
                    space=weight.input_space,
                    for_model=self.for_model,
                )
                for weight in model_weights
            ]
        )

        task = AlignModelToSpaceTask(
            tensors_base=base_weights,
            tensors_model=model_weights,
            input_transforms=input_transforms,
        )
        return {"transform": task}

    def execute(self, transform: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return transform
