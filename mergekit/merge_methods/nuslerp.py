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

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference, rectify_embed_sizes
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods.base import ConfigParameterDef, MergeMethod


class NuSlerpTask(Task[torch.Tensor]):
    gather_tensors: GatherTensors
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    parameter_name: str
    row_wise: bool

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> Tensor:
        if len(tensors) == 1:
            return list(tensors.values())[0]

        keys = list(tensors.keys())
        tensors = [tensors[key] for key in keys]
        weights = [self.tensor_parameters[key]["weight"] for key in keys]

        if len(tensors) != 2:
            raise RuntimeError("NuSlerp merge expects exactly two models")

        t = weights[1] / sum(weights)
        if abs(sum(weights)) < 1e-6:
            # this is fairly arbitrary, but it's more sane than exploding
            t = 0.5

        rectify_embed_sizes(self.parameter_name, tensors)
        return nuslerp(t, tensors[0], tensors[1], dim=0 if self.row_wise else -1)


class NuSlerpMerge(MergeMethod):
    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(
                name="nuslerp_row_wise",
                required=False,
                default_value=False,
            )
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="weight", required=True)]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: GatherTensors,
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        parameters: ImmutableMap[str, Any],
        **_kwargs,
    ) -> Task:
        return NuSlerpTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            parameter_name=output_weight.name,
            row_wise=parameters["nuslerp_row_wise"],
        )


def nuslerp(
    t: float,
    v0: torch.Tensor,
    v1: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
):
    def _normalize(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        return x / torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)

    if dim != -1:
        v0 = v0.transpose(dim, -1)
        v1 = v1.transpose(dim, -1)

    v0_u = _normalize(v0)
    v1_u = _normalize(v1)

    cos_theta = torch.sum(v0_u * v1_u, dim=-1, keepdim=True)
    theta = torch.acos(cos_theta.clamp(-1, 1))
    sin_theta = torch.sin(theta)

    colinear = (sin_theta.abs() < eps).squeeze()

    res = (torch.sin((1 - t) * theta) * v0 + torch.sin(t * theta) * v1) / sin_theta
    # Use linear interpolation for (nearly) colinear vectors
    res[colinear] = (1 - t) * v0[colinear] + t * v1[colinear]

    if dim != -1:
        res = res.transpose(dim, -1)
    return res
