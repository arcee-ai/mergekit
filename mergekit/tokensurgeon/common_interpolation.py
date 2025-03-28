# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import enum
import logging
from typing import Optional, Tuple

import torch

LOG = logging.getLogger(__name__)


class DistanceMetric(enum.Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"


class WeightingScheme(enum.Enum):
    DISTANCE_PROPORTIONAL = "distance_proportional"
    BARYCENTRIC = "barycentric"
    LEAST_SQUARES = "least_squares"


def approximate_from_landmarks(
    targets: torch.Tensor,
    points: torch.Tensor,
    distances: torch.Tensor,
    scheme: WeightingScheme = WeightingScheme.DISTANCE_PROPORTIONAL,
    cosine_similarity: bool = False,
) -> torch.Tensor:
    batch_size, embedding_dim = targets.shape
    assert points.dim() == 3 and points.shape == (
        batch_size,
        points.shape[1],
        embedding_dim,
    )
    num_points = points.shape[1]
    assert points.shape[2] == embedding_dim
    assert distances.shape == (batch_size, num_points)

    if scheme == WeightingScheme.DISTANCE_PROPORTIONAL:
        if cosine_similarity:
            weights = 1 - distances
        else:
            weights = 1 / distances.clamp_min(1e-6)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    elif scheme == WeightingScheme.BARYCENTRIC:
        weights = barycentric_weights(targets, points)
    elif scheme == WeightingScheme.LEAST_SQUARES:
        weights = torch.linalg.lstsq(
            points.transpose(1, 2).float(), targets.unsqueeze(-1).float()
        ).solution.squeeze(-1)
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    return weights


def barycentric_weights(targets: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    batch_size, num_points, _embedding_dim = points.shape
    ptp = torch.bmm(points, points.transpose(1, 2))
    ones_col = torch.ones((batch_size, num_points, 1), device=points.device)
    ones_row = torch.ones((batch_size, 1, num_points), device=points.device)
    zeros = torch.zeros((batch_size, 1, 1), device=points.device)
    upper = torch.cat([ptp, ones_col], dim=2)
    lower = torch.cat([ones_row, zeros], dim=2)
    augmented_matrix = torch.cat([upper, lower], dim=1)
    rhs_upper = torch.bmm(targets.unsqueeze(1), points.transpose(1, 2)).squeeze(1)
    rhs_lower = torch.ones((batch_size, 1), device=points.device)
    rhs = torch.cat([rhs_upper, rhs_lower], dim=1)
    return torch.linalg.lstsq(augmented_matrix, rhs.unsqueeze(-1)).solution.squeeze(-1)[
        ..., :num_points
    ]


def _cosine_sim(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def common_interp_approximate(
    targets: torch.Tensor,
    a_embeddings: torch.Tensor,
    k: Optional[int] = None,
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
    weight_scheme: WeightingScheme = WeightingScheme.DISTANCE_PROPORTIONAL,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    assert targets.dim() == 2
    assert a_embeddings.dim() == 2
    assert targets.size(1) == a_embeddings.size(1)
    assert (k is None) or (k > 0), "k must be positive"

    if metric == DistanceMetric.EUCLIDEAN:
        distances = torch.cdist(targets, a_embeddings, p=2)
    elif metric == DistanceMetric.COSINE:
        distances = 1 - _cosine_sim(targets, a_embeddings)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    # Find the k nearest neighbors
    if k is not None:
        _, indices = torch.topk(distances, k=k, dim=1, largest=False)
        knn_distances = distances.gather(1, indices)
    else:
        indices = torch.arange(a_embeddings.size(0), device=a_embeddings.device).expand(
            targets.size(0), -1
        )
        knn_distances = distances

    weights = approximate_from_landmarks(
        targets,
        a_embeddings[indices],
        knn_distances,
        scheme=weight_scheme,
        cosine_similarity=metric == DistanceMetric.COSINE,
    )

    # Log reconstruction error
    approx = (
        torch.bmm(weights.unsqueeze(1).float(), a_embeddings[indices].float())
        .squeeze(1)
        .to(targets.dtype)
    )
    err = (approx - targets).norm(dim=1)
    LOG.debug(f"Reconstruction error: {err.mean()}")
    return indices, weights
