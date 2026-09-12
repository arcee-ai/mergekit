# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import Optional

import torch

from mergekit.merge_methods.base import BasePolicy, InputContract, TensorGroup
from mergekit.merge_methods.easy_define import merge_method
from mergekit.merge_methods.generalized_task_arithmetic import (
    get_mask as sign_consensus_mask,
)


@merge_method(
    name="sce",
    pretty_name="SCE",
    reference_url="https://arxiv.org/abs/2408.07990",
    contract=InputContract(base=BasePolicy.REQUIRED),
)
def sce_merge(
    group: TensorGroup,
    int8_mask: bool = False,
    select_topk: float = 1.0,
) -> torch.Tensor:
    tensors = [entry.tensor for entry in group.non_base]
    base_tensor = group.base.tensor
    if not tensors:
        return base_tensor
    mask_dtype = torch.int8 if int8_mask else base_tensor.dtype
    task_vectors = torch.stack([t - base_tensor for t in tensors], dim=0)

    if select_topk < 1:
        mask = sce_mask(task_vectors, select_topk, mask_dtype)
        task_vectors = task_vectors * mask.unsqueeze(0)

    erase_mask = sign_consensus_mask(task_vectors, method="sum", mask_dtype=mask_dtype)

    tv_weights = sce_weight(task_vectors)
    while tv_weights.dim() < task_vectors.dim():
        tv_weights = tv_weights.unsqueeze(-1)

    erased_weights = tv_weights * erase_mask
    merged_tv = (task_vectors * erased_weights).sum(dim=0)
    final_tv = merged_tv / torch.sum(erased_weights, dim=0).clamp(min=1e-6)

    return base_tensor + final_tv


def sce_weight(tvs: torch.Tensor) -> torch.Tensor:
    weights = tvs.square().reshape(tvs.shape[0], -1).mean(dim=1)
    weight_sum = torch.sum(weights).item()
    if abs(weight_sum) < 1e-6:
        return torch.ones_like(weights) / weights.shape[0]
    return weights / weight_sum


def sce_mask(
    tvs: torch.Tensor, density: float, mask_dtype: Optional[torch.dtype] = None
):
    # The selection mask covers weight elements, never the leading input axis.
    if density <= 0:
        return torch.zeros_like(tvs[0], dtype=mask_dtype)
    if density >= 1:
        return torch.ones_like(tvs[0], dtype=mask_dtype)

    var = torch.var(tvs, dim=0, unbiased=False)
    nonzero = torch.count_nonzero(var)
    k = int(nonzero * density)
    if k == 0:
        return torch.zeros_like(var, dtype=mask_dtype)

    _, indices = torch.topk(var.abs().view(-1), k=k, largest=True)
    mask = torch.zeros_like(var, dtype=mask_dtype)
    mask.view(-1)[indices] = 1
    return mask
