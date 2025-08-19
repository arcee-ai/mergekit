# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import Any, Dict, List, Optional, Union

import torch
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


class SlerpTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    base_model: ModelReference
    t: float
    weight_info: WeightInfo

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        if len(tensors) == 1:
            return list(tensors.values())[0]
        elif len(tensors) != 2:
            raise RuntimeError("Slerp merge expects exactly two models")
        elif self.base_model not in tensors:
            raise RuntimeError("Base model not in input tensors")

        [a, b] = list(tensors.items())
        if a[0] != self.base_model:
            [a, b] = [b, a]
        prepped_tensors = [a[1], b[1]]

        rectify_embed_sizes(self.weight_info, prepped_tensors)

        return (
            slerp(
                self.t,
                prepped_tensors[0],
                prepped_tensors[1],
            )
            .to(prepped_tensors[0].dtype)
            .to(prepped_tensors[0].device)
        )

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class SlerpMerge(MergeMethod):
    def name(self) -> str:
        return "slerp"

    @override
    def pretty_name(self) -> Optional[str]:
        return "SLERP"

    @override
    def reference_url(self):
        return "https://en.wikipedia.org/wiki/Slerp"

    def parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="t", required=True)]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        base_model: Optional[ModelReference],
        **_kwargs,
    ) -> Task:
        return SlerpTask(
            gather_tensors=tensors,
            base_model=base_model,
            weight_info=output_weight,
            t=parameters["t"],
        )


def lerp(
    t: float, v0: Union[torch.tensor, torch.Tensor], v1: Union[torch.tensor, torch.Tensor]
) -> Union[torch.tensor, torch.Tensor]:
    return (1 - t) * v0 + t * v1


def slerp(
    t: Union[float, torch.tensor],
    v0: Union[torch.tensor, torch.Tensor],
    v1: Union[torch.tensor, torch.Tensor],
    DOT_THRESHOLD: float = 0.9995,
    eps: float = 1e-8,
):
    """
    Spherical linear interpolation

    From: https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    Args:
        t (float/torch.tensor): Float value between 0.0 and 1.0
        v0 (torch.tensor): Starting vector
        v1 (torch.tensor): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colinear. Not recommended to alter this.
    Returns:
        v2 (torch.tensor): Interpolation vector between v0 and v1
    """
    # Copy the vectors to reuse them later
    v0_copy = torch.clone(v0)
    v1_copy = torch.clone(v1)

    # Normalize the vectors to get the directions and angles
    v0 = normalize(v0, eps)
    v1 = normalize(v1, eps)

    # Dot product with the normalized vectors (can't use torch.dot in W)
    dot = torch.sum(v0 * v1)

    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    if torch.abs(dot) > DOT_THRESHOLD:
        res = lerp(t, v0_copy, v1_copy)
        return res

    # Calculate initial angle between v0 and v1
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)

    # Finish the slerp algorithm
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * v0_copy + s1 * v1_copy

    return res


def normalize(v: torch.tensor, eps: float):
    norm_v = torch.linalg.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v
