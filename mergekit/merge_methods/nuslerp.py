# Partly Copyright (C) 2025 Arcee AI
# Partly Copyright (C) 2025 Allura-org
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

from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from torch._tensor import Tensor
from typing_extensions import override

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


class SlerpType(str, Enum):
    nuslerp = "nuslerp"
    slerp = "slerp"


class NuSlerpTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    weight_info: WeightInfo
    row_wise: bool
    flatten: bool
    slerp_type: SlerpType
    base_model: Optional[ModelReference]
    t: Optional[float]

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> Tensor:
        if len(tensors) == 1:
            return list(tensors.values())[0]

        if self.base_model is not None:
            if self.slerp_type == SlerpType.nuslerp:
                base_tensor = tensors.pop(self.base_model)
                if len(tensors) != 2:
                    raise RuntimeError(
                        "NuSLERP merge expects exactly two models (plus *optional* base model)"
                    )
            elif self.slerp_type == SlerpType.slerp:
                if len(tensors) != 2:
                    raise RuntimeError(
                        "SLERP merge expects exactly one model (plus *required* base model)"
                    )
                base_tensor = None  # we handle this case as two regular tensors
        else:
            if self.slerp_type == SlerpType.slerp:
                raise RuntimeError("SLERP merge type requires a base model")
            base_tensor = None

        keys = list(tensors.keys())
        tensors = [tensors[key] for key in keys]
        if self.slerp_type == SlerpType.nuslerp:
            weights = [self.tensor_parameters[key]["weight"] for key in keys]
        elif self.slerp_type == SlerpType.slerp:
            weights = [
                1 - self.t,
                self.t,
            ]  # we should only be able to get to this point if t is set, don't worry abt the Optional[] type

        if abs(sum(weights)) < 1e-6:
            # this is fairly arbitrary, but it's more sane than exploding
            t = 0.5
        else:
            t = weights[1] / sum(weights)

        if base_tensor is not None:
            tensors.append(base_tensor)
        rectify_embed_sizes(self.weight_info, tensors)

        if base_tensor is not None:
            base_tensor = tensors.pop()
            return base_tensor + nuslerp(
                t,
                tensors[0] - base_tensor,
                tensors[1] - base_tensor,
                dim=0 if self.row_wise else -1,
                flatten=self.flatten,
            )
        return nuslerp(
            t,
            tensors[0],
            tensors[1],
            dim=0 if self.row_wise else -1,
            flatten=self.flatten,
        )


class NuSlerpMerge(MergeMethod):
    def name(self) -> str:
        return "slerp"

    @override
    def pretty_name(self):
        return "SLERP"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="t", required=False),
            ConfigParameterDef(
                name="nuslerp_row_wise",
                required=False,
                default_value=False,
            ),
            ConfigParameterDef(
                name="nuslerp_flatten",
                required=False,
                default_value=True,
            ),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="weight", required=False)]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **_kwargs,
    ) -> Task:
        has_t = parameters["t"] is not None
        has_weight = (
            tensor_parameters[list(tensor_parameters.keys())[0]]["weight"] is not None
        )
        if not has_t and not has_weight:
            raise RuntimeError(
                "SLERP/NuSLERP merge expects at least one model with a weight or a t parameter"
            )

        return NuSlerpTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            weight_info=output_weight,
            t=parameters["t"],
            row_wise=parameters["nuslerp_row_wise"],
            flatten=parameters["nuslerp_flatten"],
            base_model=base_model,
            slerp_type=SlerpType.slerp if has_t else SlerpType.nuslerp,
        )


def nuslerp(
    t: float,
    v0: torch.Tensor,
    v1: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    flatten: bool = False,
):
    out_shape = v0.shape

    def _normalize(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        return x / torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)

    if flatten:
        v0 = v0.view(-1)
        v1 = v1.view(-1)
    elif dim != -1:
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

    if dim != -1 and not flatten:
        res = res.transpose(dim, -1)
    return res.view(out_shape)
