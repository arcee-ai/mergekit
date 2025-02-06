# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from enum import Enum
from typing import Optional

import torch


class SparsificationMethod(str, Enum):
    magnitude = "magnitude"
    random = "random"
    magnitude_outliers = "magnitude_outliers"
    rank_magnitude_sampling = "rank_magnitude_sampling"


class RescaleNorm(str, Enum):
    l1 = "l1"
    l2 = "l2"
    linf = "linf"


def rescaled_masked_tensor(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    norm: Optional[RescaleNorm],
    eps: float = 1e-7,
) -> torch.Tensor:
    """Apply a mask to a tensor and rescale to match the original tensor norm.

    Args:
        tensor (torch.Tensor): Input tensor.
        mask (torch.Tensor): Mask to apply.
        norm (RescaleNorm): Which norm to match (l1, l2, linf).
        eps (float): Tolerance for small norms to avoid division by zero.
    """
    masked = tensor * mask
    if norm is None:
        return masked
    elif norm == RescaleNorm.l1:
        before_scale = tensor.abs().sum()
        after_scale = masked.abs().sum()
    elif norm == RescaleNorm.l2:
        before_scale = tensor.norm()
        after_scale = masked.norm()
    elif norm == RescaleNorm.linf:
        before_scale = tensor.abs().max()
        after_scale = masked.abs().max()
    else:
        raise NotImplementedError(norm)
    if before_scale < eps or after_scale < eps:
        return masked
    return masked * (before_scale / after_scale)


def magnitude(
    tensor: torch.Tensor, density: float, rescale_norm: Optional[RescaleNorm] = None
) -> torch.Tensor:
    """Masks out the smallest values, retaining a proportion of `density`."""
    if density >= 1:
        return tensor

    k = int(density * tensor.numel())

    assert k > 0, "not gonna zero out the whole tensor buddy"
    mask = torch.zeros_like(tensor)
    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    topk = torch.argsort(w, descending=True)[:k]
    mask.view(-1)[topk] = 1

    res = rescaled_masked_tensor(tensor, mask, rescale_norm)
    return res


def magnitude_outliers(
    tensor: torch.Tensor,
    density: float,
    rescale_norm: Optional[RescaleNorm] = None,
    gamma: float = 0.01,
):
    """Masks out smallest values in addition to large outliers.

    The `gamma` proportion of the largest weights are first removed, then the
    smallest weights are removed to achieve the desired density.

    Args:
        tensor (torch.Tensor): The tensor to sparsify.
        density (float): The proportion of weights to retain.
        gamma (float): Percent of largest weights to remove.
    """
    if density >= 1:
        return tensor

    num_elems = tensor.numel()
    target_n = int(density * num_elems)
    n_top = int(gamma * num_elems)
    n_bot = num_elems - target_n - n_top

    if n_bot < 0:
        # cut down on the number of large weights to remove in
        # order to hit the target density
        n_top += n_bot
        n_bot = 0

    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    indices = torch.sort(w, descending=False).indices
    mask = torch.zeros_like(tensor)

    mask.view(-1)[indices[n_bot:-n_top]] = 1

    res = rescaled_masked_tensor(tensor, mask, rescale_norm)
    return res


def bernoulli(
    tensor: torch.Tensor, density: float, rescale_norm: Optional[RescaleNorm] = None
) -> torch.Tensor:
    if density >= 1:
        return tensor

    if (tensor.device.type != "cpu") or tensor.dtype == torch.bfloat16:
        work_dtype = tensor.dtype
    else:
        # torch.bernoulli not implemented for float16 on CPU, upcast to float32
        work_dtype = torch.float32

    mask = torch.bernoulli(
        torch.full_like(input=tensor, fill_value=density, dtype=work_dtype)
    )
    res = rescaled_masked_tensor(tensor.to(work_dtype), mask, rescale_norm)
    return res.to(tensor.dtype)


def rank_magnitude(
    tensor: torch.Tensor,
    density: float,
    rescale_norm: Optional[RescaleNorm] = RescaleNorm.l1,
    epsilon: float = 0.05,
) -> torch.Tensor:
    if density >= 1:
        return tensor

    if density <= epsilon or density >= (1 - epsilon):
        raise ValueError(
            f"Error: density +- epsilon must be in the range (0, 1). density + epsilon = {density+epsilon}, density - epsilon = {density-epsilon}"
        )

    if (tensor.device.type != "cpu") or tensor.dtype == torch.bfloat16:
        work_dtype = tensor.dtype
    else:
        work_dtype = torch.float32

    if len(tensor.shape) < 2:
        tensor = tensor.unsqueeze(0)

    # Get Rank matrix for the delta values
    tensor_abs = torch.abs(tensor)

    sorted_indices = torch.argsort(tensor_abs, dim=1, descending=False)

    ranking_tensor = torch.zeros_like(tensor_abs, dtype=work_dtype)
    for i in range(tensor_abs.size(0)):
        ranking_tensor[i][sorted_indices[i]] = torch.arange(
            1, tensor.size(1) + 1, dtype=work_dtype
        ).to(tensor.device)

    # Normalise rank matrix to the probability range to density +- epsilon
    range_vals = (
        ranking_tensor.max(dim=1, keepdim=True).values
        - ranking_tensor.min(dim=1, keepdim=True).values
    )
    norm_metrics = (ranking_tensor - ranking_tensor.min(dim=1, keepdim=True).values) / (
        range_vals
    )
    final_probabilities = (density - epsilon) + norm_metrics * (2 * epsilon)

    mask = torch.bernoulli(final_probabilities).to(work_dtype)

    res = rescaled_masked_tensor(tensor.to(work_dtype), mask, rescale_norm)
    return res.squeeze(0)


def sparsify(
    tensor: torch.Tensor,
    density: float,
    method: SparsificationMethod,
    gamma: float = 0,
    rescale_norm: Optional[RescaleNorm] = None,
    epsilon: float = 0.15,
) -> torch.Tensor:
    if method == SparsificationMethod.magnitude:
        return magnitude(tensor, density=density, rescale_norm=rescale_norm)
    elif method == SparsificationMethod.random:
        return bernoulli(tensor, density=density, rescale_norm=rescale_norm)
    elif method == SparsificationMethod.magnitude_outliers:
        return magnitude_outliers(
            tensor, density=density, rescale_norm=rescale_norm, gamma=gamma
        )
    elif method == SparsificationMethod.rank_magnitude_sampling:
        return rank_magnitude(
            tensor, density=density, rescale_norm=rescale_norm, epsilon=epsilon
        )
    else:
        raise NotImplementedError(method)
