# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
from typing import Optional, Tuple

import torch

from .rope_helpers import apply_rope, estimate_pos_id_best

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


def batch_mp_resets(
    targets: torch.Tensor,
    candidate_points: torch.Tensor,
    k: int,
    eps: float = 1e-8,
    total_iterations: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """
    Matching Pursuit with Resets

    Algorithm:
    1. first perform k iterations of standard matching pursuit
    2. then, for each excess iteration, select a random index from the
       and remove it from the set of selected indices
    3. select a new one from the remaining candidates (may be the same, may be different)
    4. repeat until total_iterations are completed
    """
    if total_iterations is None:
        total_iterations = k * 3
    if total_iterations < k:
        raise ValueError(
            f"total_iterations {total_iterations} must be greater than or equal to k {k}"
        )
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
    targets_work = targets.to(dtype=work_dtype)
    points_work = candidate_points.to(dtype=work_dtype)
    selected_indices = torch.zeros((B, k), dtype=torch.long, device=device)
    mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    coeff = torch.zeros((B, k), dtype=work_dtype, device=device)
    residuals = targets_work.clone()

    iter_indices = list(range(k))
    while len(iter_indices) < total_iterations:
        honk = torch.randperm(k).tolist()
        iter_indices.extend(honk)
    iter_indices = iter_indices[:total_iterations]

    for step, t in enumerate(iter_indices):
        if step < k:
            # Initial phase: select a new candidate for position t
            inner_products = torch.matmul(residuals, points_work.T)  # B x N
            # Mask already selected points
            inner_products = inner_products.masked_fill(mask, -float("inf"))
            max_values, max_indices = torch.max(inner_products, dim=1)  # B, B
            selected_points = points_work[max_indices]  # B x D
            norms_sq = torch.sum(selected_points**2, dim=1) + eps  # B
            coeffs = max_values / norms_sq  # B
            residuals -= coeffs.unsqueeze(-1) * selected_points
            selected_indices[:, t] = max_indices
            coeff[:, t] = coeffs
            mask.scatter_(1, max_indices.unsqueeze(1), True)
        else:
            # Replacement phase: replace the candidate at position t
            old_indices = selected_indices[:, t]
            old_coeffs = coeff[:, t]
            old_points = points_work[old_indices]
            # Add back the old contribution
            residuals += old_coeffs.unsqueeze(-1) * old_points
            # Remove old index from mask
            mask.scatter_(1, old_indices.unsqueeze(1), False)
            # Compute new inner products
            inner_products = torch.matmul(residuals, points_work.T)
            inner_products = inner_products.masked_fill(mask, -float("inf"))
            new_max_values, new_max_indices = torch.max(inner_products, dim=1)
            new_points = points_work[new_max_indices]
            norms_sq = torch.sum(new_points**2, dim=1) + eps
            new_coeffs = new_max_values / norms_sq
            residuals -= new_coeffs.unsqueeze(-1) * new_points
            selected_indices[:, t] = new_max_indices
            coeff[:, t] = new_coeffs
            # Update mask with new index
            mask.scatter_(1, new_max_indices.unsqueeze(1), True)

    return selected_indices, coeff


def batch_mp_rope(
    targets: torch.Tensor,
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    k: int,
    num_heads_a: int,
    num_heads_b: int,
    eps: float = 1e-8,
    a_rope_base: float = 10000.0,
    b_rope_base: float = 10000.0,
    final_least_squares: bool = True,
) -> torch.Tensor:
    B, D_a = targets.shape
    N, _ = points_a.shape
    _, D_b = points_b.shape
    assert (
        points_a.shape[0] == points_b.shape[0]
    ), "Number of points in A and B must match"
    device = targets.device
    if k > N:
        raise ValueError(f"Cannot select {k} points from {N} candidates")
    work_dtype = (
        targets.dtype
        if targets.dtype in (torch.float32, torch.float64)
        else torch.float32
    )
    out_dtype = targets.dtype
    points_a = points_a.to(dtype=work_dtype)
    points_b = points_b.to(dtype=work_dtype)
    targets = targets.to(dtype=work_dtype)
    selected_indices = torch.zeros((B, k), dtype=torch.long, device=device)
    coeffs = torch.zeros((B, k), dtype=work_dtype, device=device)
    pos_ids = torch.zeros((B, k), dtype=work_dtype, device=device)
    mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    residuals = targets.clone()

    for t in range(k):
        abs_inner = (residuals @ points_a.T).abs()  # (B, N)
        abs_inner.masked_fill_(mask, -float("inf"))

        # Select new index with maximum correlation
        _, new_idx = torch.max(abs_inner, dim=1)  # (B,)

        # update state
        selected_indices[:, t] = new_idx
        mask[torch.arange(B, device=device), new_idx] = True
        new_atom = points_a[new_idx]

        # compute position id for new atom
        pos_id = estimate_pos_id_best(
            new_atom,
            residuals,
            num_heads=num_heads_a,
            head_dim=D_a // num_heads_a,
            base=a_rope_base,
        ).squeeze(-1)
        pos_id_neg = estimate_pos_id_best(
            new_atom,
            -residuals,
            num_heads=num_heads_a,
            head_dim=D_a // num_heads_a,
            base=a_rope_base,
        ).squeeze(-1)
        pos_id = torch.where(
            torch.abs(pos_id) < torch.abs(pos_id_neg), pos_id, pos_id_neg
        )
        pos_ids[:, t] = pos_id
        new_atom = apply_rope(
            new_atom,
            pos_id.unsqueeze(-1),
            num_heads=num_heads_a,
            head_dim=D_a // num_heads_a,
            base=a_rope_base,
        )

        # compute coefficients
        current_coeff = (residuals * new_atom).sum(dim=1) / (
            new_atom.pow(2).sum(dim=1).clamp(min=eps)
        )
        coeffs[:, t] = current_coeff

        # update residuals
        residuals = residuals - current_coeff.unsqueeze(1) * new_atom

    if final_least_squares:
        # Least-squares solve for coefficients given selected points
        # and position ids
        roped_pts_a = apply_rope(
            points_a[selected_indices],
            pos_ids.unsqueeze(-1),
            num_heads=num_heads_a,
            head_dim=D_a // num_heads_a,
            base=a_rope_base,
        )
        coeffs = torch.linalg.lstsq(
            roped_pts_a.transpose(1, 2),
            targets.unsqueeze(-1),
        ).solution.squeeze(-1)

    # return result in b space
    selected_points_b = points_b[selected_indices]
    atoms_b = apply_rope(
        selected_points_b,
        pos_ids.unsqueeze(-1),
        num_heads=num_heads_b,
        head_dim=D_b // num_heads_b,
        base=b_rope_base,
    )
    approx_b = (atoms_b * coeffs.unsqueeze(-1)).sum(dim=1)
    final_tensor = approx_b.to(out_dtype)
    return selected_indices, coeffs, final_tensor, targets - residuals
