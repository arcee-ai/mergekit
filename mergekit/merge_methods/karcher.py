# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import torch

from mergekit.merge_methods.base import TensorGroup
from mergekit.merge_methods.easy_define import merge_method


def karcher_merge_tensors(tensors, alphas, max_iter=10, tol=1e-5):
    """Compute a weighted Karcher mean on the unit hypersphere."""
    if len(tensors) == 1:
        return tensors[0]

    dtype = torch.float64 if tensors[0].dtype == torch.float64 else torch.float32
    norms = []
    units = []
    for tensor in tensors:
        norm = torch.linalg.vector_norm(tensor.to(dtype))
        # Norms remain tensors wherever they affect the result. Scalar extraction
        # is only for choosing branches and deciding when to stop iterating.
        norms.append(norm)
        if norm.item() == 0.0:
            units.append(torch.zeros_like(tensor))
        else:
            units.append((tensor / norm).to(tensor.dtype))

    valid_indices = [idx for idx, norm in enumerate(norms) if norm.item() > tol]
    if not valid_indices:
        # The constant-zero branch still has zero derivatives for every input.
        return sum(tensor * 0 for tensor in tensors)

    valid_alphas = [alphas[idx] for idx in valid_indices]
    alpha_sum = sum(valid_alphas)
    normalized_alphas = [alpha / alpha_sum for alpha in valid_alphas]
    valid_units = [units[idx] for idx in valid_indices]

    mean = torch.zeros_like(valid_units[0])
    for alpha, unit in zip(normalized_alphas, valid_units):
        mean += alpha * unit
    mean_norm = torch.linalg.vector_norm(mean.to(dtype))
    mean = (
        valid_units[0].clone()
        if mean_norm.item() < tol
        else (mean / mean_norm).to(mean.dtype)
    )

    for _ in range(max_iter):
        tangent = torch.zeros_like(mean)
        for alpha, unit in zip(normalized_alphas, valid_units):
            dot = torch.clamp(torch.dot(mean.flatten(), unit.flatten()), -1.0, 1.0)
            theta = torch.arccos(dot)
            if theta.item() < tol:
                continue
            tangent += alpha * (theta / torch.sin(theta)) * (unit - dot * mean)

        tangent_norm = torch.linalg.vector_norm(tangent.to(dtype))
        if tangent_norm.item() < tol:
            break
        mean = (
            torch.cos(tangent_norm) * mean
            + torch.sin(tangent_norm) * (tangent / tangent_norm)
        ).to(mean.dtype)
        unit_norm = torch.linalg.vector_norm(mean.to(dtype))
        if unit_norm.item() > tol:
            mean = (mean / unit_norm).to(mean.dtype)

    scale = sum(alpha * norm for alpha, norm in zip(alphas, norms))
    return scale * mean


@merge_method(
    name="karcher",
    pretty_name="Karcher Mean",
    reference_url="https://arxiv.org/abs/2603.04972",
)
def karcher_merge_method(
    group: TensorGroup,
    max_iter: int = 10,
    tol: float = 1e-5,
) -> torch.Tensor:
    tensors = [entry.tensor for entry in group.entries]
    if len(tensors) == 1:
        return tensors[0]
    alphas = [1.0 / len(tensors)] * len(tensors)
    return karcher_merge_tensors(tensors, alphas, max_iter=max_iter, tol=tol)
