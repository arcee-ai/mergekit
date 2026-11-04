# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from enum import Enum
from typing import Optional

import torch


class SparsificationMethod(str, Enum):
    magnitude = "magnitude"
    random = "random"
    magnitude_outliers = "magnitude_outliers"
    della_magprune = "della_magprune"


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


def _smallest_k_mask(w: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean mask selecting exactly ``k`` of the smallest entries of ``w``.

    ``w`` must be a non-negative 1-D CPU tensor without NaNs. Runs in linear
    time instead of sorting. Entries equal to the k-th smallest value are taken
    in index order, so the result always has exactly ``k`` elements and the
    mask for ``j <= k`` is a subset of the mask for ``k``.
    """
    n = w.numel()
    if k <= 0:
        return torch.zeros(n, dtype=torch.bool, device=w.device)
    if k >= n:
        return torch.ones(n, dtype=torch.bool, device=w.device)
    if w.dtype in (torch.float16, torch.bfloat16):
        # Non-negative floats order the same way as their bit patterns, so a
        # histogram over the 2**15 possible values gives an exact selection.
        # This also avoids quickselect, which slows down on the heavily
        # repeated values that half-precision deltas contain.
        keys = w.view(torch.int16).to(torch.int32)
        cumulative = torch.bincount(keys, minlength=1 << 15).cumsum(0)
        threshold = torch.searchsorted(cumulative, torch.tensor(k))
        mask = keys < threshold
        ties = keys == threshold
    else:
        threshold = torch.kthvalue(w, k).values
        mask = w < threshold
        ties = w == threshold
    remaining = k - int(mask.sum())
    if remaining > 0:
        mask[torch.nonzero(ties).squeeze(1)[:remaining]] = True
    return mask


def _use_fast_selection(w: torch.Tensor) -> bool:
    # Linear-time selection is much faster than sorting on CPU. Other devices
    # keep the sort-based path, and NaNs (which sort as largest) also fall
    # back so the selected set is unchanged.
    return (
        w.device.type == "cpu"
        and w.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        and not bool(torch.isnan(w).any())
    )


def magnitude(
    tensor: torch.Tensor, density: float, rescale_norm: Optional[RescaleNorm] = None
) -> torch.Tensor:
    """Masks out the smallest values, retaining a proportion of `density`.

    Positive density retains at least one element in nonempty tensors.
    """
    if density >= 1:
        return tensor
    if density <= 0 or tensor.numel() == 0:
        return torch.zeros_like(tensor)

    k = max(1, int(density * tensor.numel()))

    w = tensor.abs().reshape(-1)
    if _use_fast_selection(w):
        # The k largest are everything except the (numel - k) smallest.
        mask = (~_smallest_k_mask(w, w.numel() - k)).to(tensor.dtype)
    else:
        if w.device.type == "cpu":
            w = w.float()
        mask = torch.zeros(tensor.numel(), dtype=tensor.dtype, device=tensor.device)
        topk = torch.argsort(w, descending=True)[:k]
        mask[topk] = 1

    res = rescaled_masked_tensor(tensor, mask.reshape_as(tensor), rescale_norm)
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
    # Reduce the outlier removal when necessary to retain the target density.
    n_bot = max(0, num_elems - target_n - n_top)

    w = tensor.abs().reshape(-1)
    if _use_fast_selection(w):
        # Keep ascending ranks [n_bot, n_bot + target_n).
        keep = _smallest_k_mask(w, n_bot + target_n) & ~_smallest_k_mask(w, n_bot)
        mask = keep.to(tensor.dtype)
    else:
        if w.device.type == "cpu":
            w = w.float()
        indices = torch.sort(w, descending=False).indices
        mask = torch.zeros(num_elems, dtype=tensor.dtype, device=tensor.device)
        mask[indices[n_bot : n_bot + target_n]] = 1

    res = rescaled_masked_tensor(tensor, mask.reshape_as(tensor), rescale_norm)
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


def della_magprune(
    tensor: torch.Tensor,
    density: float,
    epsilon: float,
    rescale_norm: Optional[RescaleNorm] = None,
) -> torch.Tensor:
    if density >= 1:
        return tensor
    if density <= 0:
        return torch.zeros_like(tensor)
    orig_shape = tensor.shape

    if density + epsilon >= 1 or density - epsilon <= 0:
        raise ValueError(
            "Epsilon must be chosen such that density +/- epsilon is in (0, 1)"
        )

    work_dtype = (
        tensor.dtype
        if tensor.device.type != "cpu" or tensor.dtype == torch.bfloat16
        else torch.float32
    )

    if len(tensor.shape) < 2:
        tensor = tensor.unsqueeze(0)
    magnitudes = tensor.abs()

    sorted_indices = torch.argsort(magnitudes, dim=1, descending=False)
    # Invert the permutation with a scatter instead of a second argsort.
    # Shape the positions as (1, n, 1, ...) so they broadcast along dim 1
    # for weights of any rank (e.g. conv or patch-embedding tensors).
    positions = (
        torch.arange(sorted_indices.shape[1], device=sorted_indices.device)
        .view(1, -1, *([1] * (sorted_indices.dim() - 2)))
        .expand_as(sorted_indices)
    )
    ranks = torch.empty_like(sorted_indices).scatter_(1, sorted_indices, positions)
    ranks = ranks.to(work_dtype) + 1

    min_ranks = ranks.min(dim=1, keepdim=True).values
    max_ranks = ranks.max(dim=1, keepdim=True).values
    rank_norm = ((ranks - min_ranks) / (max_ranks - min_ranks)).clamp(0, 1)
    probs = (density - epsilon) + rank_norm * 2 * epsilon
    mask = torch.bernoulli(probs).to(work_dtype)

    res = rescaled_masked_tensor(tensor.to(work_dtype), mask, rescale_norm)
    return res.to(tensor.dtype).reshape(orig_shape)


def sparsify(
    tensor: torch.Tensor,
    density: float,
    method: SparsificationMethod,
    gamma: float = 0,
    epsilon: float = 0,
    rescale_norm: Optional[RescaleNorm] = None,
) -> torch.Tensor:
    if method == SparsificationMethod.magnitude:
        return magnitude(tensor, density=density, rescale_norm=rescale_norm)
    elif method == SparsificationMethod.random:
        return bernoulli(tensor, density=density, rescale_norm=rescale_norm)
    elif method == SparsificationMethod.magnitude_outliers:
        return magnitude_outliers(
            tensor, density=density, rescale_norm=rescale_norm, gamma=gamma
        )
    elif method == SparsificationMethod.della_magprune:
        return della_magprune(
            tensor, density=density, epsilon=epsilon, rescale_norm=rescale_norm
        )
    else:
        raise NotImplementedError(method)
