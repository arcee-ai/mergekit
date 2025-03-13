# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import List, Optional

import torch

from mergekit.merge_methods.easy_define import merge_method


@merge_method(
    name="multislerp",
    pretty_name="Multi-SLERP",
    reference_url="https://goddard.blog/posts/multislerp-wow-what-a-cool-idea",
)
def multislerp(
    tensors: List[torch.Tensor],
    weight: List[float],
    base_tensor: Optional[torch.Tensor] = None,
    normalize_weights: bool = True,
    eps: float = 1e-8,
):
    """
    Implements barycentric interpolation on a hypersphere.

    The approach:
    1. Project points onto a tangent space at their weighted Euclidean mean.
    2. Perform the interpolation in the tangent space.
    3. Project the result back to the hypersphere.

    Limitations:
    - The weighted sum of the input tensors must not be zero.
    - The tensors must not be all parallel or antiparallel.

    Args:
        tensors: List of tensors to interpolate
        weight: List of weights for each tensor
        base_tensor: Optional tensor defining the origin of the hypersphere
        normalize_weights: If True, the weights will be normalized to sum to 1
        eps: Small constant for numerical stability
    """
    if len(tensors) == 1:
        # No interpolation needed
        return tensors[0]

    tensors = torch.stack(tensors, dim=0)
    if base_tensor is not None:
        tensors -= base_tensor

    tensors_flat = tensors.view(tensors.shape[0], -1)

    weights = torch.tensor(weight, dtype=tensors.dtype, device=tensors.device)
    if normalize_weights:
        weights = weights / weights.sum()

    # Project to unit hypersphere
    norms = torch.norm(tensors_flat, dim=-1, keepdim=True)
    unit_tensors = tensors_flat / (norms + eps)

    mean = (unit_tensors * weights.view(-1, 1)).sum(0)
    mean_norm = torch.norm(mean)
    if mean_norm < eps:
        if tensors.shape[0] == 2:
            # fallback to linear interpolation
            res = (tensors[0] * weights[0] + tensors[1] * weights[1]).view(
                tensors.shape[1:]
            )
            if base_tensor is not None:
                res = res + base_tensor
            return res
        raise ValueError(
            "The weighted sum of the input tensors is zero. This occurs when "
            "antipodal vectors or sets of vectors have weights that exactly "
            "balance out (e.g., vectors a,-a with equal weights). Try using "
            "different weights if you have antipodal vectors."
        )
    mean = mean / mean_norm

    # Project to tangent space
    dots = (unit_tensors * mean).sum(-1, keepdim=True)
    tangent_vectors = unit_tensors - dots * mean

    # Interpolate
    tangent_result = (tangent_vectors * weights.view(-1, 1)).sum(0)

    # Project back to sphere using exponential map
    tangent_norm = torch.norm(tangent_result) + eps
    result = mean * torch.cos(tangent_norm) + tangent_result * (
        torch.sin(tangent_norm) / tangent_norm
    )

    avg_norm = (norms.squeeze(-1) * weights).sum()
    result = result * avg_norm
    result = result.view(tensors.shape[1:])

    if base_tensor is not None:
        result = result + base_tensor

    return result
