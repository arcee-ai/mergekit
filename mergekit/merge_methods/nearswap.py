# Copyright (C) 2025 Arcee AI
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

from typing import Any, Dict, List, Optional

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
        if self.t <= 0:
            raise RuntimeError(f"Threshold cannot be <= zero, got {self.t}")
        if len(tensors) == 1:
            return list(tensors.values())[0]
        elif len(tensors) != 2:
            raise RuntimeError(
                f"Nearswap merge expects exactly two models, got {len(tensors)}"
            )
        elif self.base_model not in tensors:
            raise RuntimeError("Base model not in input tensors")

        [a, b] = list(tensors.items())
        if a[0] != self.base_model:
            [a, b] = [b, a]
        prepped_tensors = [a[1], b[1]]

        rectify_embed_sizes(self.weight_info, prepped_tensors)

        return (
            nearswap(
                self.t,
                prepped_tensors[0],
                prepped_tensors[1],
            )
            .to(prepped_tensors[0].dtype)
            .to(prepped_tensors[0].device)
        )


class NearSwapMerge(MergeMethod):
    def name(self) -> str:
        return "nearswap"

    def pretty_name(self) -> Optional[str]:
        return "NearSwap"

    def reference_url(self) -> Optional[str]:
        return "https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001"

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


def nearswap(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """
    NearSwap implementation using PyTorch.

    Adapted from: https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001

    Parameters:
        t (float): The sameness threshold.
        v0 (torch.Tensor): Weights from the base model.
        v1 (torch.Tensor): Weights from the secondary model.

    Returns:
        torch.Tensor: Resulting interpolated weights.
    """
    # Compute the absolute difference
    lweight = torch.abs(v0 - v1)

    # Compute the interpolation factor
    lweight = t / lweight
    lweight = torch.nan_to_num(lweight, nan=1.0, posinf=1.0, neginf=1.0)
    lweight = torch.clamp(lweight, min=0.0, max=1.0)

    # Linearly interpolate between v0 and v1
    return lweight * v1 + (1 - lweight) * v0
