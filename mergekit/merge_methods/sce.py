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

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel
from typing_extensions import override

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)


class SCEMerge(MergeMethod, BaseModel, frozen=True):
    def name(self) -> str:
        return "sce"

    @override
    def pretty_name(self) -> str:
        return "SCE"

    @override
    def reference_url(self) -> str:
        return "https://arxiv.org/abs/2408.07990"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="int8_mask", required=False, default_value=False),
            ConfigParameterDef(name="select_topk", required=False, default_value=1.0),
        ]

    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        **_kwargs,
    ) -> Task:
        return SCETask(
            tensors=tensors,
            base_model=base_model,
            int8_mask=parameters["int8_mask"],
            select_topk=parameters["select_topk"],
            weight_info=output_weight,
        )


class SCETask(Task[torch.Tensor]):
    tensors: MergeTensorInput
    base_model: ModelReference
    weight_info: WeightInfo
    int8_mask: bool
    select_topk: float

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.tensors}

    def execute(
        self,
        tensors: Dict[ModelReference, torch.Tensor],
        **_kwargs,
    ) -> torch.Tensor:
        # collect task vectors
        tvs, base = get_task_vectors(self.weight_info, self.base_model, tensors)
        if not tvs:
            return base

        deltas = torch.stack([tv["delta"] for tv in tvs], dim=0)
        mask_dtype = torch.int8 if self.int8_mask else base.dtype

        # Select the top τ% elements with high variance
        if self.select_topk < 1:
            mask = get_sce_mask(deltas, self.select_topk, mask_dtype)
            mask_expanded = mask.unsqueeze(0).expand_as(deltas)
            deltas = deltas * mask_expanded

        # Calculate matrix level merging coefficient
        weights = get_sce_weight(deltas)
        weights = torch.tensor(weights, dtype=deltas.dtype, device=deltas.device)
        while len(deltas.shape) > len(weights.shape):
            weights.unsqueeze_(-1)

        # Erase elements with minority directions
        erase_mask = get_erase_mask(
            deltas,
            mask_dtype=mask_dtype,
        )
        erased_weights = weights * erase_mask
        mixed_delta = (deltas * erased_weights).sum(dim=0)

        # Normalize
        divisor = (erased_weights).sum(dim=0)
        divisor[divisor == 0] = 1
        mixed_delta /= divisor

        return (base + mixed_delta).to(base.dtype)


def get_task_vectors(
    weight_info: WeightInfo,
    base_model: ModelReference,
    tensors: ImmutableMap[ModelReference, torch.Tensor],
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    keys = list(tensors.keys())
    base = tensors[base_model]

    parameter_name = weight_info.name

    res = []
    for model in keys:
        if model == base_model:
            continue

        x = tensors[model].to(base.dtype)
        if x.shape != base.shape:
            if weight_info.is_embed:
                x = x[: base.shape[0], : base.shape[1]]
                logging.warning(f"Using submatrix of {model}:{parameter_name}")
            else:
                logging.warning(
                    f"skipping {model}:{parameter_name} due to size mismatch"
                )
                continue

        delta = x - base
        del x
        del tensors[model]

        d = {}
        d["model"] = model
        d["delta"] = delta
        res.append(d)
    return res, base


def get_erase_mask(
    delta: torch.Tensor,
    mask_dtype: Optional[torch.dtype] = None,
):
    """Returns a mask determining which delta vectors should be merged
    into the final model.
    """
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    sign_weight = delta.sum(dim=0)
    majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
    del sign_weight

    return sign == majority_sign


def get_sce_mask(
    deltas: torch.Tensor,
    density: float,
    mask_dtype: Optional[torch.dtype] = None,
):
    if mask_dtype is None:
        mask_dtype = deltas.dtype
    # Calculate variance along the first dimension
    variance = torch.var(deltas, dim=0, unbiased=False)
    # Count non-zero positions in variance
    non_zero_positions_count = torch.count_nonzero(variance)
    # Calculate the number of top elements to select
    k = int(abs(density) * non_zero_positions_count)
    mask = torch.zeros_like(variance, dtype=mask_dtype)
    if k == 0:
        return mask
    assert k > 0, "not gonna zero out the whole tensor buddy"

    # Get the indices of the top k elements with the highest absolute variance
    topk_indices = torch.topk(variance.abs().view(-1), k=k, largest=True).indices

    mask.view(-1)[topk_indices] = 1
    return mask


def get_sce_weight(deltas):
    # Calculate the squared sum of each delta and normalize by the number of elements
    weights = [torch.sum(delta**2).item() / delta.numel() for delta in deltas]

    # Normalize the weights
    sum_weights = sum(weights)
    if sum_weights == 0:
        weights = [1.0 / len(weights)] * len(weights)
    else:
        weights = [w / sum_weights for w in weights]

    return weights
