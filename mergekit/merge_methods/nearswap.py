# Copyright (C) 2025 Gordon Freeman
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

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


class NearSwapTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    base_model: ModelReference
    t: float
    weight_info: WeightInfo

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        if self.t == 0:
            raise RuntimeError("Threshold cannot be zero")
        if len(tensors) == 1:
            return list(tensors.values())[0]
        elif len(tensors) != 2:
            raise RuntimeError("Nearswap merge expects exactly two models")
        elif self.base_model not in tensors:
            raise RuntimeError("Base model not in input tensors")

        [a, b] = list(tensors.items())
        if a[0] != self.base_model:
            [a, b] = [b, a]
        prepped_tensors = [a[1], b[1]]

        rectify_embed_sizes(self.weight_info, prepped_tensors)

        return (
            NearSwap(
                self.t,
                prepped_tensors[0],
                prepped_tensors[1],
            )
            .to(prepped_tensors[0].dtype)
            .to(prepped_tensors[0].device)
        )


class NearSwapMerge(MergeMethod):
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
        return NearSwapTask(
            gather_tensors=tensors,
            base_model=base_model,
            weight_info=output_weight,
            t=parameters["t"],
        )


def lerp(
    t: float, v0: Union[np.ndarray, torch.Tensor], v1: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    return (1 - t) * v0 + t * v1


def NearSwap(
    t: float,
    v0: Union[np.ndarray, torch.Tensor],
    v1: Union[np.ndarray, torch.Tensor],
):
    """
    Interpolates base model with secondary model if the distance between base and secondary model is below t

    From: https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001
    Args:
        t (float/np.ndarray): Non zero float
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
    Returns:
        v2 (np.ndarray): Model with interpolated weights below t
    """
    is_torch = False
    if not isinstance(v0, np.ndarray):
        is_torch = True
        v0 = v0.detach().cpu().float().numpy()
    if not isinstance(v1, np.ndarray):
        is_torch = True
        v1 = v1.detach().cpu().float().numpy()
    lweight = np.absolute(v0 - v1)
    lweight = np.nan_to_num(lweight, nan=1.0, posinf=1.0, neginf=1.0)
    np.clip(lweight, a_min=0.0, a_max=1.0, out=lweight)
    res = lerp(lweight, v0, v1)

    return maybe_torch(res, is_torch)


def maybe_torch(v: np.ndarray, is_torch: bool):
    if is_torch:
        return torch.from_numpy(v)
    return v
