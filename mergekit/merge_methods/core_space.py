from typing import List

import torch
from torch import Tensor

from mergekit.merge_methods.easy_define import merge_method


def _lora_factor_from_delta(delta: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    """
    Approximate a LoRA-style factorization ΔW ≈ B @ A
    delta: [m, n]
    returns: A: [r, n], B: [m, r] such that B@A ~ delta
    """
    # SVD in float32 for numerical stability
    U, S, Vh = torch.linalg.svd(delta.to(torch.float32), full_matrices=False)
    r_eff = min(rank, S.shape[0])
    if r_eff == 0:
        # No meaningful update at all
        m, n = delta.shape
        return delta.new_zeros((0, n)), delta.new_zeros((m, 0))

    U_r = U[:, :r_eff]  # [m, r_eff]
    S_r = S[:r_eff]  # [r_eff]
    Vh_r = Vh[:r_eff, :]  # [r_eff, n]

    # Δ ≈ (U * sqrt(S)) @ (sqrt(S) * Vh)
    sqrt_S = S_r.sqrt()
    B = U_r * sqrt_S.unsqueeze(0)  # scale columns of U
    A = sqrt_S.unsqueeze(1) * Vh_r  # scale rows of Vh
    return A, B


def _core_space_ta_single(
    tensors: List[Tensor],
    base_tensor: Tensor,
    rank: int,
) -> Tensor:
    """
    Core Space Task Arithmetic for a single 2D weight matrix.
    tensors: list of [m, n] (full weights for each model)
    base_tensor: [m, n]
    """
    if not tensors:
        return base_tensor

    # Compute per-task deltas
    deltas = [t.to(torch.float32) - base_tensor.to(torch.float32) for t in tensors]

    # Factorize each delta as Δ ≈ B @ A
    A_list: List[Tensor] = []
    B_list: List[Tensor] = []
    for delta in deltas:
        if delta.abs().max() == 0:
            # pure base, ignore
            continue
        A, B = _lora_factor_from_delta(delta, rank=rank)
        if A.numel() == 0 or B.numel() == 0:
            continue
        A_list.append(A)  # [r, n]
        B_list.append(B)  # [m, r]

    if not A_list:
        # no non‑zero updates
        return base_tensor

    m, n = base_tensor.shape
    T = len(A_list)
    # All A_i are [r, n], B_i are [m, r]
    r = A_list[0].shape[0]

    # Stack to build global reference bases (Core Space)
    A_stack = torch.cat(A_list, dim=0)  # [T * r, n]
    B_stack = torch.cat(B_list, dim=1)  # [m, T * r]

    # SVDs to get reference bases
    # A_stack: we want a "right" basis for columns -> Vh
    _, _, Vh_A_ref = torch.linalg.svd(A_stack, full_matrices=False)  # [R_A, n]
    # B_stack: we want a "left" basis for rows -> U
    U_B_ref, _, _ = torch.linalg.svd(B_stack, full_matrices=False)  # [m, R_B]

    # Make both bases the same core dimensionality R
    R = min(U_B_ref.shape[1], Vh_A_ref.shape[0])
    U_B_ref = U_B_ref[:, :R]  # [m, R]
    Vh_A_ref = Vh_A_ref[:R, :]  # [R, n]

    # Represent each task in Core Space:
    # M_t = U_B_ref^T @ B_t @ A_t @ Vh_A_ref^T
    M_list: List[Tensor] = []
    for A, B in zip(A_list, B_list):
        # shapes: [R, m] @ [m, r] @ [r, n] @ [n, R] -> [R, R]
        M_aligned = U_B_ref.T @ B @ A @ Vh_A_ref.T
        M_list.append(M_aligned)

    # Task Arithmetic in Core Space: sum the core matrices
    M_merged = torch.stack(M_list, dim=0).sum(dim=0)  # [R, R]

    # Reconstruct merged delta in the original weight space:
    # Δ_merged = U_B_ref @ M_merged @ Vh_A_ref
    delta_merged = U_B_ref @ M_merged @ Vh_A_ref  # [m, n]
    return (base_tensor.to(torch.float32) + delta_merged).to(base_tensor.dtype)


@merge_method(
    name="core_ta",
    pretty_name="Task Arithmetic in Core Space",
    reference_url="https://arxiv.org/abs/2509.17786",
)
def core_space_task_arithmetic(
    tensors: List[Tensor],
    base_tensor: Tensor,
    rank: int = 16,
) -> Tensor:
    """
    Merge method: 'core_ta'

    - For 1D / scalar tensors: falls back to simple Task Arithmetic.
    - For 2D tensors: run Core Space Task Arithmetic as in Panariello et al. (NeurIPS 2025).
    - 'rank' controls the truncation rank when approximating LoRA-style factors.
    """
    if not tensors:
        return base_tensor

    # Non-matrix parameters: simple Task Arithmetic (sum of deltas)
    if base_tensor.ndim < 2:
        deltas = torch.stack(
            [t.to(torch.float32) - base_tensor.to(torch.float32) for t in tensors],
            dim=0,
        )
        merged = base_tensor.to(torch.float32) + deltas.sum(dim=0)
        return merged.to(base_tensor.dtype)

    # Treat last two dims as the matrix, keep any leading dims (e.g. heads)
    *prefix, m, n = base_tensor.shape
    if len(prefix) == 0:
        # Simple [m, n] case
        return _core_space_ta_single(tensors, base_tensor, rank=rank)

    # Handle batched / multihead weights by looping over the prefix
    base_2d = base_tensor.reshape(-1, m, n)
    merged_2d = torch.empty_like(base_2d)
    tensor_2d_list = [t.reshape(-1, m, n) for t in tensors]

    for i in range(base_2d.shape[0]):
        layer_base = base_2d[i]
        layer_tensors = [t[i] for t in tensor_2d_list]
        merged_2d[i] = _core_space_ta_single(layer_tensors, layer_base, rank=rank)

    return merged_2d.reshape(base_tensor.shape)
