# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
from typing import Tuple

import torch

LOG = logging.getLogger(__name__)


def batch_omp(
    targets: torch.Tensor,
    candidate_points: torch.Tensor,
    k: int,
    eps: float = 1e-8,
    reorthogonalize_interval: int = 256,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """
    Batched Orthogonal Matching Pursuit (OMP) to select `k` points from `candidate_points` that best approximate each target in `targets`.

    Args:
        targets: (B, D) tensor of target vectors.
        candidate_points: (N, D) tensor of candidate points.
        k: Number of points to select (sparsity level).
        eps: Tolerance for numerical stability.
        reorthogonalize_interval: Number of iterations between reorthogonalization steps.

    Returns:
        selected_indices: (B, k) tensor of indices selected for each target.
        coeff: (B, k) tensor of coefficients for each selected point.
    """
    B, D = targets.shape
    N, _ = candidate_points.shape
    device = targets.device
    if k > N:
        raise ValueError(f"Cannot select {k} points from {N} candidates")
    work_dtype = (
        targets.dtype
        if targets.dtype in (torch.float32, torch.float64)
        else torch.float32
    )
    # Convert inputs to work_dtype
    targets_work = targets.to(dtype=work_dtype)
    points_work = candidate_points.to(dtype=work_dtype)
    # Preallocate tensors
    q = torch.zeros((B, D, k), dtype=work_dtype, device=device)
    r = torch.zeros((B, k, k), dtype=work_dtype, device=device)
    selected_indices = torch.zeros((B, k), dtype=torch.long, device=device)
    mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    residuals = targets_work.clone()

    for t in range(k):
        rms_0 = residuals.norm(dim=1).mean()
        # Compute absolute inner products between residuals and points
        abs_inner = (residuals @ points_work.T).abs()  # (B, N)
        # Mask out already selected points
        abs_inner.masked_fill_(mask, -float("inf"))

        # Select new index with maximum correlation
        _, new_idx = torch.max(abs_inner, dim=1)  # (B,)
        selected_indices[:, t] = new_idx
        mask[torch.arange(B, device=device), new_idx] = True

        new_atom = points_work[new_idx]  # (B, D)
        if t == 0:
            r[:, 0, 0] = new_atom.norm(dim=1)
            norm = r[:, 0, 0].clamp(min=eps)
            q[:, :, 0] = new_atom / norm.unsqueeze(1)
        else:
            # Project onto existing basis
            projections = torch.bmm(
                q[:, :, :t].transpose(1, 2), new_atom.unsqueeze(-1)
            ).squeeze(
                -1
            )  # (B, t)
            residual = new_atom - torch.bmm(
                q[:, :, :t], projections.unsqueeze(-1)
            ).squeeze(
                -1
            )  # (B, D)
            norm = torch.clamp(torch.norm(residual, dim=1), min=eps)
            # Update R and Q
            r[:, :t, t] = projections
            r[:, t, t] = norm
            q[:, :, t] = residual / norm.unsqueeze(-1)

        if t > 0 and t % reorthogonalize_interval == 0:
            q_b = q[:, :, : t + 1]
            q_new, r_new = torch.linalg.qr(q_b, mode="reduced")
            r[:, : t + 1, : t + 1] = torch.bmm(r_new, r[:, : t + 1, : t + 1])
            q[:, :, : t + 1] = q_new

        qt_targets = torch.bmm(
            q[:, :, : t + 1].transpose(1, 2), targets_work.unsqueeze(-1)
        )  # (B, t+1, 1)
        approx = torch.bmm(q[:, :, : t + 1], qt_targets).squeeze(-1)
        residuals = targets_work - approx
        LOG.debug(f"OMP iteration {t}: RMS {rms_0} -> {residuals.norm(dim=1).mean()}")

    # Get final coefficients
    final_coeff = torch.linalg.solve_triangular(
        r[:, :k, :k],
        torch.bmm(q[:, :, :k].transpose(1, 2), targets_work.unsqueeze(-1)),
        upper=True,
    ).squeeze(-1)

    # Print residuals if we're yapping
    if LOG.isEnabledFor(logging.DEBUG):
        rt_approx = torch.bmm(
            final_coeff.unsqueeze(1), points_work[selected_indices]
        ).squeeze(1)
        residuals = targets_work - rt_approx
        LOG.debug(f"OMP final RMS: {residuals.norm(dim=1).mean()}")

    return selected_indices, final_coeff
