import pytest
import torch

from mergekit.sparsify import SparsificationMethod, sparsify


@pytest.fixture
def sample_tensor():
    res = torch.randn(128, 64)
    res[res == 0] = 7  # very low chance, but hey!
    return res


class TestMagnitude:
    def test_full_density(self, sample_tensor):
        assert torch.equal(
            sparsify(sample_tensor, density=1, method=SparsificationMethod.magnitude),
            sample_tensor,
        )

    def test_zero_density(self, sample_tensor):
        with pytest.raises(AssertionError):
            sparsify(sample_tensor, density=0, method=SparsificationMethod.magnitude)

    def test_partial_density(self, sample_tensor):
        result = sparsify(
            sample_tensor, density=0.5, method=SparsificationMethod.magnitude
        )
        assert torch.count_nonzero(result) == sample_tensor.view(-1).shape[0] // 2

    def test_outliers(self, sample_tensor):
        for gamma_0 in [0.1, 0.2, 0.5, 1.0]:
            for density in [0.1, 0.3, 0.5, 0.6, 0.9, 1.0]:
                sparsity = 1 - density
                gamma = gamma_0 * sparsity
                result = sparsify(
                    sample_tensor,
                    density=density,
                    method=SparsificationMethod.magnitude_outliers,
                    gamma=gamma,
                )
                assert torch.count_nonzero(result) == int(
                    sample_tensor.view(-1).shape[0] * density
                )

    def test_norm_rescale(self, sample_tensor):
        l1_norm = sample_tensor.abs().sum()
        l2_norm = sample_tensor.norm()
        linf_norm = sample_tensor.abs().max()

        normed_l1 = sparsify(
            sample_tensor,
            density=0.5,
            method=SparsificationMethod.magnitude,
            rescale_norm="l1",
        )
        normed_l2 = sparsify(
            sample_tensor,
            density=0.5,
            method=SparsificationMethod.magnitude,
            rescale_norm="l2",
        )
        normed_linf = sparsify(
            sample_tensor,
            density=0.5,
            method=SparsificationMethod.magnitude,
            rescale_norm="linf",
        )

        assert torch.isclose(normed_l1.abs().sum(), l1_norm, rtol=0.01)
        assert torch.isclose(normed_l2.norm(), l2_norm, rtol=0.01)
        assert torch.isclose(normed_linf.abs().max(), linf_norm, rtol=0.01)

    def test_della_magprune(self, sample_tensor):
        res = sparsify(
            sample_tensor,
            density=0.5,
            method=SparsificationMethod.della_magprune,
            epsilon=0.05,
            rescale_norm="l1",
        )
        assert not res.isnan().any(), "NaNs in result tensor"
        assert not res.isinf().any(), "Infs in result tensor"


class TestBernoulli:
    NUM_ITERATIONS = 1000

    def test_bernoulli_with_rescale(self, sample_tensor):
        ref_abs_sum = sample_tensor.abs().sum()
        avg_abs_sum = torch.zeros_like(ref_abs_sum)
        for _ in range(TestBernoulli.NUM_ITERATIONS):
            rescaled = sparsify(
                sample_tensor,
                density=0.5,
                method=SparsificationMethod.random,
                rescale_norm="l1",
            )
            avg_abs_sum += rescaled.abs().sum()
        avg_abs_sum /= TestBernoulli.NUM_ITERATIONS

        assert torch.isclose(avg_abs_sum, ref_abs_sum, rtol=0.01)

    def test_bernoulli_without_rescale(self, sample_tensor):
        result = sparsify(
            sample_tensor,
            density=0.5,
            method=SparsificationMethod.random,
            rescale_norm=None,
        )
        assert 0 < torch.count_nonzero(result) <= sample_tensor.view(-1).shape[0]

    def test_cpu_dtypes(self, sample_tensor):
        for dt in (torch.float16, torch.bfloat16, torch.float32):
            sparsify(
                tensor=sample_tensor.to(dtype=dt).cpu(),
                density=0.5,
                method=SparsificationMethod.random,
                rescale_norm="l1",
            )
