# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging

import torch

LOG = logging.getLogger(__name__)


def landmark_pca_approximate(
    targets: torch.Tensor,
    points_a: torch.Tensor,
    points_b: torch.Tensor,
) -> torch.Tensor:
    """Given target points in space a and a set of reference points in both space a and b,
    approximate the target points in space b."""
    # points_a: (N, D_a)
    # points_b: (N, D_b)
    # 1:1 correspondence between points_a and points_b
    # targets: (B, D_a)
    num_points, d_a = points_a.shape
    batch_size, _ = targets.shape
    _, d_b = points_b.shape
    assert (
        points_a.shape[0] == points_b.shape[0]
    ), "Number of points in A and B must match"
    assert targets.shape == (batch_size, d_a)

    effective_dim = min(d_a, d_b)

    out_dtype = targets.dtype
    points_a = points_a.float()
    points_b = points_b.float()
    targets = targets.float()

    # Compute the mean of all points in A and B
    mean_a = points_a.mean(dim=0, keepdim=True)  # (1, D_a)
    mean_b = points_b.mean(dim=0, keepdim=True)  # (1, D_b)
    centered_a = points_a - mean_a  # (N, D_a)
    centered_b = points_b - mean_b  # (N, D_b)
    centered_targets = targets - mean_a  # (B, D_a)

    # Perform PCA to get the principal components
    U_a, S_a, V_a = torch.pca_lowrank(centered_a, q=effective_dim)
    U_b, S_b, V_b = torch.pca_lowrank(centered_b, q=effective_dim)

    # Project reference points into PCA space
    A_pca = torch.mm(centered_a, V_a)  # (N, effective_dim)
    B_pca = torch.mm(centered_b, V_b)  # (N, effective_dim)

    # Compute Procrustes matrix and solve for optimal rotation
    M = torch.mm(B_pca.t(), A_pca)  # (effective_dim, effective_dim)
    U, S, V = torch.svd(M)
    R = torch.mm(U, V.t())  # (effective_dim, effective_dim)

    # Transform targets through PCA spaces and rotation
    projected_a = torch.mm(centered_targets, V_a)  # (B, effective_dim)
    rotated = torch.mm(projected_a, R)  # (B, effective_dim)
    projected_b = torch.mm(rotated, V_b.t())  # (B, D_b)

    # Translate back to original space B
    approximated_b = projected_b + mean_b

    return approximated_b.to(out_dtype)
