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

from typing import Dict, Generic, List, Optional, Tuple, Union

import torch
from scipy.optimize import linear_sum_assignment
from torch._tensor import Tensor
from typing_extensions import TypeVar

from mergekit.architecture import ProceduralSpaceInfo, WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import LoadTensor

CollectKeyT = TypeVar("CollectKeyT")
CollectValueT = TypeVar("CollectValueT")


class NullTask(Task[None]):
    def execute(self) -> None:
        return None

    def arguments(self):
        return {}


class CollectTupleTask(Task[Tuple]):
    """Collects the results of a number of tasks into a tuple."""

    tasks: Tuple[Task, ...]

    def arguments(self) -> Dict[str, Task]:
        return {f"a_{idx}": task for idx, task in enumerate(self.tasks)}

    def execute(self, **kwargs) -> List:
        return tuple([kwargs[f"a_{idx}"] for idx in range(len(self.tasks))])

    def group_label(self) -> Optional[str]:
        return max(t.group_label() or "" for t in self.arguments().values())


class CollectDictTask(
    Task[Dict[CollectKeyT, CollectValueT]], Generic[CollectKeyT, CollectValueT]
):
    """Collects a dictionary of tasks into a dictionary of results."""

    tasks: ImmutableMap[CollectKeyT, Task[CollectValueT]]

    @property
    def _ordered_keys(self):
        return list(sorted(self.tasks.keys(), key=repr))

    def arguments(self) -> Dict[str, Task]:
        return {
            "elements": CollectTupleTask(
                tasks=[self.tasks[key] for key in self._ordered_keys]
            )
        }

    def execute(
        self, elements: List[CollectValueT]
    ) -> Dict[CollectKeyT, CollectValueT]:
        return dict(zip(self._ordered_keys, elements))

    def group_label(self) -> Optional[str]:
        return max(t.group_label() or "" for t in self.arguments().values())


class AlignModelToSpaceTask(Task[torch.Tensor]):
    tensors_base: Tuple[Task, ...]
    tensors_model: Tuple[Task, ...]
    input_transforms: Optional[Tuple[Optional[Task], ...]] = None

    def arguments(self) -> Dict[str, Task]:
        res = {}
        if self.input_transforms:
            res["input_transforms"] = CollectTupleTask(tasks=self.input_transforms)
        res["base_tensors"] = CollectTupleTask(tasks=self.tensors_base)
        res["model_tensors"] = CollectTupleTask(tasks=self.tensors_model)
        return res

    def execute(
        self,
        base_tensors: List[torch.Tensor],
        model_tensors: List[torch.Tensor],
        input_transforms: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tensor:
        out_dim = base_tensors[0].shape[0]
        if not all(t.shape[0] == out_dim for t in (base_tensors + model_tensors)):
            raise RuntimeError(
                "All tensors that share an output space must have same output dimension"
            )

        work_dtype = (
            torch.float32
            if base_tensors[0].device.type == "cpu"
            else base_tensors[0].dtype
        )

        if input_transforms:
            # Apply input transformations to model tensors
            new_model_tensors = []
            for x_model, tf_in in zip(model_tensors, input_transforms):
                if tf_in is not None:
                    new_model_tensors.append(
                        x_model.to(work_dtype) @ tf_in.to(work_dtype)
                    )
                else:
                    new_model_tensors.append(x_model)
            model_tensors = new_model_tensors

        # Solve LAP to find best permutation of model weights to base weights
        cost_mat = torch.zeros(
            out_dim, out_dim, device=base_tensors[0].device, dtype=work_dtype
        )
        for x_base, x_model in zip(base_tensors, model_tensors):
            cost_mat += x_base.to(work_dtype) @ x_model.T.to(work_dtype)

        ri, ci = linear_sum_assignment(cost_mat.numpy(), maximize=True)
        model_to_base = torch.zeros_like(cost_mat, dtype=base_tensors[0].dtype)
        model_to_base[(ri, ci)] = 1

        return model_to_base


class TransposeTensor(Task[Optional[torch.Tensor]]):
    tensor_task: Task

    def arguments(self) -> Dict[str, Task]:
        return {
            "tensor": self.tensor_task,
        }

    def execute(
        self,
        tensor: Optional[torch.Tensor],
    ) -> Optional[Tensor]:
        if tensor is None:
            return None
        return tensor.T


class GetAlignedTensor(Task[Optional[torch.Tensor]]):
    tensor_task: Union[Task[torch.Tensor], Task[Optional[torch.Tensor]]]
    transform_in: Union[Task[Optional[torch.Tensor]], Task[torch.Tensor], None] = None
    transform_out: Union[Task[Optional[torch.Tensor]], Task[torch.Tensor], None] = None

    def arguments(self) -> Dict[str, Task]:
        return {
            "tensor": self.tensor_task,
            "transform_in": self.transform_in,
            "transform_out": self.transform_out,
        }

    def execute(
        self,
        tensor: Optional[torch.Tensor],
        transform_in: Optional[torch.Tensor] = None,
        transform_out: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None

        work_dtype = torch.float32 if tensor.device.type == "cpu" else tensor.dtype
        out_dtype = tensor.dtype

        if transform_in is not None:
            tensor = tensor.to(dtype=work_dtype) @ transform_in.to(dtype=work_dtype)
        if transform_out is not None:
            tensor = transform_out.to(dtype=work_dtype) @ tensor.to(dtype=work_dtype)
        return tensor.to(dtype=out_dtype)


class ResidualSpaceTransform(Task[torch.Tensor]):
    input_transform_tasks: Tuple[Task, ...]

    def arguments(self) -> Dict[str, Task]:
        return {"transforms": CollectTupleTask(tasks=self.input_transform_tasks)}

    def execute(self, transforms: List[torch.Tensor]) -> Tensor:
        valid = [t for t in transforms if t is not None]
        if not valid:
            return None
        return sum(valid) / max(len(valid), 1)


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

    def align_tensor(
        self, model: ModelReference, weight: WeightInfo, tensor_task: Task
    ) -> Task:
        if weight.is_embed:
            tensor_task = TransposeTensor(tensor_task=tensor_task)
        res = GetAlignedTensor(
            tensor_task=tensor_task,
            transform_out=(
                DelayedAlignTask(
                    planner=self,
                    space=weight.output_space,
                    for_model=model,
                )
                if weight.output_space
                else None
            ),
            transform_in=(
                TransposeTensor(
                    tensor_task=DelayedAlignTask(
                        planner=self,
                        space=weight.input_space,
                        for_model=model,
                    )
                )
                if weight.input_space
                else None
            ),
        )
        if weight.is_embed:
            # and untranspose
            res = TransposeTensor(tensor_task=res)
        return res


class DelayedAlignTask(Task[Optional[torch.Tensor]], arbitrary_types_allowed=True):
    planner: SpacePlanner
    space: str
    for_model: ModelReference
    dtype: Optional[str] = None

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
                (
                    DelayedAlignTask(
                        planner=self.planner,
                        space=weight.input_space,
                        for_model=self.for_model,
                    )
                    if weight.input_space is not None
                    else NullTask()
                )
                for weight in model_weights
            ]
        )

        def _load(model: ModelReference, w: WeightInfo):
            res = LoadTensor(
                model=model,
                tensor=w.name,
                dtype=self.dtype,
                optional=w.optional,
                aliases=w.aliases,
            )
            if w.is_embed:
                # embeddings store weights with shape (vocab_size, embed_dim)
                # so flip 'em to (embed_dim, vocab_size)
                res = TransposeTensor(tensor_task=res)
            return res

        task = AlignModelToSpaceTask(
            tensors_base=tuple(_load(self.planner.base_model, w) for w in base_weights),
            tensors_model=tuple(_load(self.for_model, w) for w in model_weights),
            input_transforms=input_transforms,
        )
        return {"transform": task}

    def execute(
        self, transform: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        return transform
