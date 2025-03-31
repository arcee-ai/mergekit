# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

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


class NuSlerpTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    weight_info: WeightInfo
    row_wise: bool
    flatten: bool
    base_model: Optional[ModelReference]

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> Tensor:
        if len(tensors) == 1:
            return list(tensors.values())[0]

        if self.base_model is not None:
            if len(tensors) != 3:
                raise RuntimeError(
                    "NuSlerp base model can not be one of the two models to merge"
                )
            base_tensor = tensors.pop(self.base_model)
        else:
            base_tensor = None

        keys = list(tensors.keys())
        tensors = [tensors[key] for key in keys]
        weights = [self.tensor_parameters[key]["weight"] for key in keys]

        if len(tensors) != 2:
            raise RuntimeError(
                "NuSlerp merge expects exactly two models (plus optional base model)"
            )

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
        return "nuslerp"

    @override
    def pretty_name(self):
        return "NuSLERP"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
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
        return [ConfigParameterDef(name="weight", required=True)]

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
        return NuSlerpTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            weight_info=output_weight,
            row_wise=parameters["nuslerp_row_wise"],
            flatten=parameters["nuslerp_flatten"],
            base_model=base_model,
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
