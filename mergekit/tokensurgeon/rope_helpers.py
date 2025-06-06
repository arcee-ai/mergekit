from typing import Optional

import torch


def llama_rope_rotationmat(theta: torch.Tensor) -> torch.Tensor:
    """
    Create a rotation matrix for RoPE as used in the `transformers` Llama implementation.

    Args:
        theta: Tensor of shape (..., n_heads, head_dim // 2) representing the angles for the rotation.
    """
    # theta shape: (..., n_heads, head_dim // 2)
    n_heads = theta.shape[-2]
    head_dim = theta.shape[-1] * 2
    theta_p = torch.cat([theta, theta], dim=-1)
    cos_theta = torch.cos(theta_p)
    sin_theta = torch.sin(theta_p)
    P = torch.zeros(
        tuple(list(theta.shape[:-1]) + [head_dim, head_dim]),
        dtype=theta.dtype,
        device=theta.device,
    )
    idx = torch.arange(head_dim // 2)
    P[..., :, idx, idx] = cos_theta[..., :, idx]
    P[..., :, idx, head_dim // 2 + idx] = sin_theta[..., :, idx]
    P[..., :, head_dim // 2 + idx, idx] = -sin_theta[..., :, idx]
    P[..., :, head_dim // 2 + idx, head_dim // 2 + idx] = cos_theta[..., :, idx]
    return P


def _rope_inv_freq(
    base: float, dim: int, device: Optional["torch.device"] = None
) -> torch.Tensor:
    return 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )


def estimate_theta(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Estimate a set of per-head, per-dimension angles (theta) such that
    rotating x_0 by theta will least-squares approximate x_1.

    Args:
        x_0: Tensor of shape (..., n_heads*head_dim) representing the first input.
        x_1: Tensor of shape (..., n_heads*head_dim) representing the second input.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
    Returns:
        Tensor of shape (..., n_heads, head_dim // 2) representing the estimated theta values.
    """
    # Reshape inputs to separate heads
    x0_reshaped = x_0.view(*x_0.shape[:-1], num_heads, head_dim)
    x1_reshaped = x_1.view(*x_1.shape[:-1], num_heads, head_dim)

    # Split into pairs of dimensions
    half_dim = head_dim // 2
    x0_i = x0_reshaped[..., :half_dim]
    x0_j = x0_reshaped[..., half_dim:]
    x1_i = x1_reshaped[..., :half_dim]
    x1_j = x1_reshaped[..., half_dim:]

    # Compute A_d and B_d for each pair
    A_d = x0_i * x1_i + x0_j * x1_j  # (..., num_heads, half_dim)
    B_d = x0_i * x1_j - x0_j * x1_i  # (..., num_heads, half_dim)

    # Compute theta using least-squares approximation
    theta = torch.atan2(B_d, A_d)  # (..., num_heads, half_dim)

    return theta


def estimate_position_id(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    num_heads: int,
    head_dim: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """
    Estimate a scalar position ID such that applying RoPE to x_0
    will least-squares approximate x_1.

    Args:
        x_0: Tensor of shape (..., n_heads*head_dim) representing the first input.
        x_1: Tensor of shape (..., n_heads*head_dim) representing the second input.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        base: Base value used for frequency computation. Defaults to 10000.0.

    Returns:
        Tensor of shape (..., 1) representing the estimated position ID.
    """
    # Reshape inputs to separate heads and head dimensions
    x0_heads = x_0.view(*x_0.shape[:-1], num_heads, head_dim)
    x1_heads = x_1.view(*x_1.shape[:-1], num_heads, head_dim)

    # Split each head's dimensions into two halves for pairs (d, d + head_dim//2)
    split_idx = head_dim // 2
    x0_a = x0_heads[..., :split_idx]  # (..., nh, hd//2)
    x0_b = x0_heads[..., split_idx:]  # (..., nh, hd//2)
    x1_c = x1_heads[..., :split_idx]
    x1_d = x1_heads[..., split_idx:]

    numerator = x0_a * x1_d - x0_b * x1_c  # (..., nh, hd//2)
    denominator = x0_a * x1_c + x0_b * x1_d
    theta = torch.arctan2(numerator, denominator)  # (..., nh, hd//2)

    inv_freq = _rope_inv_freq(base, head_dim, x_0.device)  # (hd//2, )
    pos_i = theta / inv_freq  # (..., nh, hd//2)
    weights = x0_a.pow(2) + x0_b.pow(2)  # (..., nh, hd//2)
    sum_pos = (pos_i * weights).sum(dim=(-1, -2))  # (...)
    sum_weights = weights.sum(dim=(-1, -2))  # (...)
    pos_estimate = sum_pos / (sum_weights + 1e-8)
    return pos_estimate.unsqueeze(-1)


def estimate_position_id_projection(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    num_heads: int,
    head_dim: int,
    base: float = 10000.0,
) -> torch.Tensor:
    inv_freq = _rope_inv_freq(base, head_dim, x_0.device)  # (head_dim//2, )
    basis_vector = inv_freq.view(1, 1, head_dim // 2).expand(
        x_0.shape[:-1] + (num_heads, head_dim // 2)
    )  # (..., n_heads, head_dim//2)
    basis_vector = basis_vector.reshape(*x_0.shape[:-1], -1)
    basis_vector_norm = torch.norm(basis_vector, dim=-1, keepdim=True)
    basis_vector = basis_vector / (basis_vector_norm + 1e-8)  # Normalize to unit length
    theta = estimate_theta(x_0, x_1, num_heads, head_dim)  # (..., n_heads, head_dim//2)
    theta = theta.reshape(*x_0.shape[:-1], -1)  # (..., n_heads*head_dim//2)
    projection = torch.sum(theta * basis_vector, dim=-1)  # (...)
    # Compute scaling factor
    f_norm = torch.norm(inv_freq)
    scaling_factor = torch.sqrt(torch.tensor(num_heads, device=x_0.device)) * f_norm
    pos_estimate = projection / (scaling_factor + 1e-8)
    return pos_estimate.unsqueeze(-1)  # (..., 1)


def apply_rope_theta(
    x: torch.Tensor,
    theta: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Apply RoPE to the input tensor x using the given theta.
    Args:
        x: Tensor of shape (..., n_heads*head_dim) representing the input.
        theta: Tensor of shape (..., n_heads, head_dim // 2) representing the angles.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
    Returns:
        Tensor of shape (..., n_heads*head_dim) representing the output.
    """
    # Reshape input to separate heads
    x_reshaped = x.view(*x.shape[:-1], num_heads, head_dim)

    # Split into two halves for rotation
    half_dim = head_dim // 2
    x_i = x_reshaped[..., :half_dim]
    x_j = x_reshaped[..., half_dim:]

    # Calculate cosine and sine of theta
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Apply rotation transformation
    x_i_rot = x_i * cos_theta - x_j * sin_theta
    x_j_rot = x_j * cos_theta + x_i * sin_theta

    # Concatenate the rotated features
    rotated = torch.cat([x_i_rot, x_j_rot], dim=-1)

    # Reshape back to original shape
    return rotated.view(*x.shape)


def estimate_pos_id_best(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    num_heads: int,
    head_dim: int,
    base: float = 10000.0,
) -> torch.Tensor:
    return estimate_position_id_projection(
        x_0,
        x_1,
        num_heads,
        head_dim,
        base=base,
    )
    est_pos_id = estimate_position_id(
        x_0,
        x_1,
        num_heads,
        head_dim,
        base=base,
    )
    est_pos_id_proj = estimate_position_id_projection(
        x_0,
        x_1,
        num_heads,
        head_dim,
        base=base,
    )
    rt_pos_id = apply_rope(
        x_0,
        est_pos_id,
        num_heads,
        head_dim,
        base=base,
    )
    rt_pos_id_proj = apply_rope(
        x_0,
        est_pos_id_proj,
        num_heads,
        head_dim,
        base=base,
    )
    err_pos_id = torch.norm(rt_pos_id - x_1, dim=-1)
    err_pos_id_proj = torch.norm(rt_pos_id_proj - x_1, dim=-1)
    res = torch.where(
        err_pos_id < err_pos_id_proj,
        est_pos_id,
        est_pos_id_proj,
    )
    return res


def apply_rope(
    x: torch.Tensor,
    pos: torch.Tensor,
    num_heads: int,
    head_dim: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """
    Apply RoPE to the input tensor x using the given position pos.
    Args:
        x: Tensor of shape (..., n_heads*head_dim) representing the input.
        pos: Tensor of shape (..., 1) representing the position ID.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        base: Base value used for frequency computation. Defaults to 10000.0.
    Returns:
        Tensor of shape (..., n_heads*head_dim) representing the output.
    """
    inv_freq = _rope_inv_freq(base, head_dim, x.device)
    theta = pos.unsqueeze(-1) * inv_freq

    return apply_rope_theta(
        x,
        theta,
        num_heads,
        head_dim,
    )
