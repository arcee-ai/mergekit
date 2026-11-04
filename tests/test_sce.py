import pytest
import torch

from mergekit.merge_methods.sce import sce_mask, sce_merge


@pytest.mark.parametrize(
    ("density", "expected_value"),
    [
        (-0.1, 0),
        (0, 0),
        (0.01, 0),
        (1, 1),
        (1.1, 1),
    ],
)
def test_sce_mask_early_exit_shape(density, expected_value):
    tvs = torch.stack((torch.zeros(2, 3), torch.ones(2, 3)))

    mask = sce_mask(tvs, density)

    assert mask.shape == tvs.shape[1:]
    assert torch.all(mask == expected_value)


def test_sce_merge_zero_density_preserves_base_shape():
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tensors = [base + 1, base - 1]

    result = sce_merge(tensors, base, select_topk=0)

    assert result.shape == base.shape
    torch.testing.assert_close(result, base)
