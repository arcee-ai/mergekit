import torch

from mergekit.tokensurgeon.pca import landmark_pca_approximate


def test_landmark_pca_maps_held_out_points():
    """Space b is a noisy linear image of space a. The Procrustes rotation
    between their principal axes must carry held-out points closer to their
    images than the mean of b is."""
    generator = torch.Generator().manual_seed(0)
    num_points, num_targets, dim_a, dim_b = 2000, 200, 32, 48
    points = torch.randn(num_points + num_targets, dim_a, generator=generator)
    points = points * torch.linspace(3.0, 0.3, dim_a)
    mixing = torch.randn(dim_a, dim_b, generator=generator) / dim_a**0.5
    images = points @ mixing + 0.05 * torch.randn(
        num_points + num_targets, dim_b, generator=generator
    )
    points_a, targets = points[:num_points], points[num_points:]
    points_b, expected = images[:num_points], images[num_points:]

    torch.manual_seed(0)
    result = landmark_pca_approximate(targets, points_a, points_b)

    error = (result - expected).norm() / (expected - points_b.mean(dim=0)).norm()
    assert error < 0.5
