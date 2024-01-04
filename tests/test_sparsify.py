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


class TestBernoulli:
    NUM_ITERATIONS = 1000

    def test_bernoulli_with_rescale(self, sample_tensor):
        ref_abs_sum = sample_tensor.abs().sum()
        avg_abs_sum = torch.zeros_like(ref_abs_sum)
        for _ in range(TestBernoulli.NUM_ITERATIONS):
            rescaled = sparsify(
                sample_tensor, density=0.5, method=SparsificationMethod.rescaled_random
            )
            avg_abs_sum += rescaled.abs().sum()
        avg_abs_sum /= TestBernoulli.NUM_ITERATIONS

        assert torch.isclose(avg_abs_sum, ref_abs_sum, rtol=0.01)

    def test_bernoulli_without_rescale(self, sample_tensor):
        result = sparsify(
            sample_tensor, density=0.5, method=SparsificationMethod.random
        )
        assert 0 < torch.count_nonzero(result) <= sample_tensor.view(-1).shape[0]

    def test_cpu_dtypes(self, sample_tensor):
        for dt in (torch.float16, torch.bfloat16, torch.float32):
            sparsify(
                tensor=sample_tensor.to(dtype=dt).cpu(),
                density=0.5,
                method=SparsificationMethod.rescaled_random,
            )
