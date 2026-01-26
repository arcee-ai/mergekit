# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import List, Tuple

import torch

from mergekit.merge_methods.easy_define import merge_method


@merge_method(
    name="ram",
    pretty_name="Reinforced Agent Merging",
    reference_url="https://arxiv.org/abs/2601.13572",
)
def ram_merge(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    if not tensors:
        return base_tensor

    (
        tv_flat,
        nonzero_mask,
        contrib_counts,
        overlap_mask,
        unique_mask,
    ) = _prepare_ram_vectors(tensors, base_tensor, epsilon)

    tv_flat_z = tv_flat * nonzero_mask
    merged_tv_flat = (
        (tv_flat_z * unique_mask)
        + (tv_flat_z * overlap_mask / contrib_counts.clamp(min=1))
    ).sum(dim=0, keepdim=True)

    return base_tensor + merged_tv_flat.reshape_as(base_tensor)


@merge_method(
    name="ramplus_tl",
    pretty_name="Reinforced Agent Merging Plus (Tensor-Local)",
    reference_url="https://arxiv.org/abs/2601.13572",
)
def ramplus_tl_merge(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    r: float = 0.1,
    alpha: float = 0.2,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    if not tensors:
        return base_tensor

    (
        tv_flat,
        nonzero_mask,
        contrib_counts,
        overlap_mask,
        unique_mask,
    ) = _prepare_ram_vectors(tensors, base_tensor, epsilon)

    shared_counts = (nonzero_mask & overlap_mask).sum(dim=1)
    unique_counts = (nonzero_mask & unique_mask).sum(dim=1)
    rho = shared_counts / unique_counts.clamp(min=epsilon)
    lambda_ = 1 + r * rho.clamp(min=0, max=alpha)

    tv_flat_z = tv_flat * nonzero_mask
    merged_tv_flat = (
        (tv_flat_z * unique_mask * lambda_.unsqueeze(-1))
        + (tv_flat_z * overlap_mask / contrib_counts.clamp(min=1))
    ).sum(dim=0, keepdim=True)
    merged_tv = merged_tv_flat.reshape_as(base_tensor)
    return base_tensor + merged_tv


def _prepare_ram_vectors(
    tensors: List[torch.Tensor], base_tensor: torch.Tensor, epsilon: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to compute task vectors, masks, and counts shared by RAM methods.
    Returns:
        tv_flat: Flattened task vectors
        nonzero_mask: Mask of values > epsilon
        contrib_counts: Count of models contributing to each parameter
        overlap_mask: Mask where counts > 1
        unique_mask: Mask where counts == 1
    """
    task_vectors = torch.stack([t - base_tensor for t in tensors], dim=0)
    tv_flat = task_vectors.view(len(tensors), -1)

    # Create masks based on epsilon threshold
    nonzero_mask = tv_flat.abs() > epsilon

    # Count how many models contribute to each specific parameter
    # shape: (1, num_params)
    contrib_counts = nonzero_mask.sum(dim=0, keepdim=True)

    overlap_mask = contrib_counts > 1
    unique_mask = contrib_counts == 1

    return tv_flat, nonzero_mask, contrib_counts, overlap_mask, unique_mask
