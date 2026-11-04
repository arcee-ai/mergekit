import torch

from mergekit.tokensurgeon.omp import batch_mp_rope
from mergekit.tokensurgeon.rope_helpers import (
    apply_rope,
    estimate_pos_id_best,
    estimate_position_id,
    estimate_position_id_projection,
)


def test_estimate_pos_id_best_selects_lower_error_candidate():
    """estimate_pos_id_best must compare both position estimators per sample
    and pick the one with smaller RoPE reconstruction error."""
    generator = torch.Generator().manual_seed(42)
    batch_size, num_heads, head_dim = 8, 4, 16
    x_0 = torch.randn(batch_size, num_heads * head_dim, generator=generator)

    # Concentrate energy in lower-frequency pairs so positions with |pos| > pi
    # wrap the high-frequency pairs (hurting projection) while remaining
    # unwrapped in the dominant lower-frequency pairs (helping weighted mean).
    x_0_heads = x_0.view(batch_size, num_heads, head_dim)
    x_0_heads[..., : head_dim // 4] *= 0.01
    x_0_heads[..., head_dim // 2 : 3 * head_dim // 4] *= 0.01

    true_pos = torch.tensor(
        [[1.5], [-2.0], [0.5], [10.0], [-5.0], [3.2], [0.8], [-1.1]]
    )
    x_1 = apply_rope(x_0, true_pos, num_heads, head_dim)

    pos_weighted = estimate_position_id(x_0, x_1, num_heads, head_dim)
    pos_proj = estimate_position_id_projection(x_0, x_1, num_heads, head_dim)
    pos_best = estimate_pos_id_best(x_0, x_1, num_heads, head_dim)

    assert pos_best.shape == (batch_size, 1)

    err_weighted = torch.norm(
        apply_rope(x_0, pos_weighted, num_heads, head_dim) - x_1, dim=-1
    )
    err_proj = torch.norm(apply_rope(x_0, pos_proj, num_heads, head_dim) - x_1, dim=-1)
    err_best = torch.norm(apply_rope(x_0, pos_best, num_heads, head_dim) - x_1, dim=-1)

    expected_min_err = torch.minimum(err_weighted, err_proj)
    assert torch.allclose(err_best, expected_min_err, atol=1e-7)
    assert (err_best < err_proj).any()


def test_batch_mp_rope_runs_with_batched_targets():
    """batch_mp_rope calls estimate_pos_id_best on batched targets and must
    preserve (B, k) shapes."""
    generator = torch.Generator().manual_seed(0)
    batch_size, num_points, k = 6, 32, 4
    num_heads_a, num_heads_b, head_dim = 4, 4, 8
    dim_a = num_heads_a * head_dim
    dim_b = num_heads_b * head_dim

    points_a = torch.randn(num_points, dim_a, generator=generator)
    points_b = torch.randn(num_points, dim_b, generator=generator)
    targets = torch.randn(batch_size, dim_a, generator=generator)

    indices, coeffs, approx_b, approx_a = batch_mp_rope(
        targets=targets,
        points_a=points_a,
        points_b=points_b,
        k=k,
        num_heads_a=num_heads_a,
        num_heads_b=num_heads_b,
    )

    assert indices.shape == (batch_size, k)
    assert coeffs.shape == (batch_size, k)
    assert approx_b.shape == (batch_size, dim_b)
    assert approx_a.shape == (batch_size, dim_a)
