import pytest
import torch

from mergekit.sparsify import SparsificationMethod, _smallest_k_mask, sparsify


@pytest.fixture
def sample_tensor():
    res = torch.randn(128, 64)
    res[res == 0] = 7  # very low chance, but hey!
    return res


class TestMagnitude:
    @pytest.mark.parametrize("gamma", [0.0, 0.01, 0.25])
    def test_strided_outlier_selection(self, gamma):
        tensor = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4).T
        result = sparsify(
            tensor,
            density=0.5,
            method=SparsificationMethod.magnitude_outliers,
            gamma=gamma,
        )
        largest_kept = 12 - int(gamma * 12)
        expected = tensor * ((tensor > largest_kept - 6) & (tensor <= largest_kept))
        torch.testing.assert_close(result, expected)

    def test_full_density(self, sample_tensor):
        assert torch.equal(
            sparsify(sample_tensor, density=1, method=SparsificationMethod.magnitude),
            sample_tensor,
        )

    @pytest.mark.parametrize("rescale_norm", [None, "l1", "l2", "linf"])
    def test_zero_density(self, sample_tensor, rescale_norm):
        result = sparsify(
            sample_tensor,
            density=0,
            method=SparsificationMethod.magnitude,
            rescale_norm=rescale_norm,
        )
        torch.testing.assert_close(result, torch.zeros_like(sample_tensor))

    @pytest.mark.parametrize("density", [0.01, 0.5])
    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    @pytest.mark.parametrize(
        "values,expected",
        [
            (3.0, 3.0),
            ([-3.0], [-3.0]),
            ([1.0, -4.0, 2.0], [0.0, -4.0, 0.0]),
            ([[1.0, -4.0, 2.0]], [[0.0, -4.0, 0.0]]),
        ],
    )
    def test_small_tensor_retains_largest_value(self, density, dtype, values, expected):
        tensor = torch.tensor(values, dtype=dtype)
        result = sparsify(tensor, density, SparsificationMethod.magnitude)
        torch.testing.assert_close(result, torch.tensor(expected, dtype=dtype))
        torch.testing.assert_close(tensor, torch.tensor(values, dtype=dtype))

    @pytest.mark.parametrize("shape", [(0,), (2, 0)])
    @pytest.mark.parametrize("rescale_norm", [None, "l1", "l2", "linf"])
    def test_empty_tensor(self, shape, rescale_norm):
        tensor = torch.empty(shape)
        result = sparsify(
            tensor,
            density=0.5,
            method=SparsificationMethod.magnitude,
            rescale_norm=rescale_norm,
        )
        torch.testing.assert_close(result, tensor)

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


def _sorted_magnitude_mask(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Reference: the original sort-based top-k selection."""
    k = max(1, int(density * tensor.numel()))
    w = tensor.abs().reshape(-1).float()
    mask = torch.zeros(tensor.numel(), dtype=torch.bool)
    mask[torch.argsort(w, descending=True)[:k]] = True
    return mask.reshape_as(tensor)


def _sorted_outlier_mask(tensor: torch.Tensor, density: float, gamma: float):
    """Reference: the original sort-based outlier selection."""
    n = tensor.numel()
    target_n = int(density * n)
    n_bot = max(0, n - target_n - int(gamma * n))
    w = tensor.abs().reshape(-1).float()
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.sort(w).indices[n_bot : n_bot + target_n]] = True
    return mask.reshape_as(tensor)


class TestFastSelection:
    """The CPU path selects with kthvalue instead of a full sort."""

    @pytest.mark.parametrize("density", [0.01, 0.1, 0.5, 0.9, 0.999])
    @pytest.mark.parametrize("shape", [(7,), (64, 33), (3, 5, 17)])
    def test_magnitude_matches_sort_without_ties(self, density, shape):
        torch.manual_seed(0)
        tensor = torch.randn(shape, dtype=torch.float64)
        assert tensor.abs().unique().numel() == tensor.numel()
        result = sparsify(tensor, density, SparsificationMethod.magnitude)
        torch.testing.assert_close(
            result, tensor * _sorted_magnitude_mask(tensor, density)
        )

    @pytest.mark.parametrize("gamma", [0.0, 0.01, 0.2])
    @pytest.mark.parametrize("density", [0.1, 0.5, 0.8])
    def test_outliers_match_sort_without_ties(self, density, gamma):
        torch.manual_seed(1)
        tensor = torch.randn(96, 40, dtype=torch.float64)
        result = sparsify(
            tensor, density, SparsificationMethod.magnitude_outliers, gamma=gamma
        )
        torch.testing.assert_close(
            result, tensor * _sorted_outlier_mask(tensor, density, gamma)
        )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("density", [0.05, 0.37, 0.5, 0.93])
    def test_magnitude_heavy_ties_keeps_exactly_k_largest(self, dtype, density):
        torch.manual_seed(2)
        # Only 4 distinct magnitudes, so nearly every threshold is a tie.
        signs = torch.randint(0, 2, (50, 41)) * 2 - 1
        tensor = (torch.randint(1, 5, (50, 41)) * signs).to(dtype)
        k = max(1, int(density * tensor.numel()))
        result = sparsify(tensor, density, SparsificationMethod.magnitude)
        kept = result != 0
        assert int(kept.sum()) == k
        kept_mag = tensor.abs()[kept].float()
        dropped_mag = tensor.abs()[~kept].float()
        if dropped_mag.numel():
            assert kept_mag.min() >= dropped_mag.max()

    @pytest.mark.parametrize("gamma", [0.0, 0.05, 0.3])
    def test_outliers_heavy_ties_keep_a_rank_window(self, gamma):
        torch.manual_seed(3)
        tensor = torch.randint(1, 8, (60, 30)).to(torch.bfloat16)
        density = 0.4
        n = tensor.numel()
        target_n = int(density * n)
        n_bot = max(0, n - target_n - int(gamma * n))
        result = sparsify(
            tensor, density, SparsificationMethod.magnitude_outliers, gamma=gamma
        )
        kept = result != 0
        assert int(kept.sum()) == target_n
        mags = tensor.abs().float().reshape(-1)
        ranked = torch.sort(mags).values
        kept_mags = mags[kept.reshape(-1)]
        # Every kept magnitude lies inside the requested ascending-rank window.
        assert kept_mags.min() >= ranked[n_bot]
        assert kept_mags.max() <= ranked[n_bot + target_n - 1]

    def test_nan_falls_back_to_sort(self):
        torch.manual_seed(4)
        tensor = torch.randn(20, 10)
        tensor[3, 4] = float("nan")
        result = sparsify(tensor, 0.3, SparsificationMethod.magnitude)
        expected = tensor * _sorted_magnitude_mask(tensor, 0.3)
        torch.testing.assert_close(result, expected, equal_nan=True)

    @pytest.mark.parametrize(
        "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64]
    )
    def test_smallest_k_mask_is_nested_and_exact(self, dtype):
        w = torch.tensor([3.0, 1.0, 2.0, 1.0, 2.0, 1.0, 5.0, 0.0, -0.0], dtype=dtype)
        w = w.abs()
        previous = torch.zeros_like(w, dtype=torch.bool)
        for k in range(w.numel() + 1):
            mask = _smallest_k_mask(w, k)
            assert int(mask.sum()) == k
            assert not (previous & ~mask).any()
            if 0 < k < w.numel():
                assert w[mask].max() <= w[~mask].min()
            previous = mask

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_della_ranks_match_double_argsort(self, dtype):
        tensor = torch.randn(33, 70).to(dtype)

        def reference(t):
            work_dtype = torch.float32 if dtype != torch.bfloat16 else dtype
            magnitudes = t.abs()
            ranks = magnitudes.argsort(dim=1).argsort(dim=1).to(work_dtype) + 1
            min_r = ranks.min(dim=1, keepdim=True).values
            max_r = ranks.max(dim=1, keepdim=True).values
            rank_norm = ((ranks - min_r) / (max_r - min_r)).clamp(0, 1)
            mask = torch.bernoulli((0.5 - 0.1) + rank_norm * 2 * 0.1).to(work_dtype)
            return (t.to(work_dtype) * mask).to(t.dtype)

        torch.manual_seed(5)
        expected = reference(tensor)
        torch.manual_seed(5)
        result = sparsify(tensor, 0.5, SparsificationMethod.della_magprune, epsilon=0.1)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("shape", [(8, 16, 3), (4, 6, 3, 3), (6, 5, 1)])
    def test_della_ranks_match_double_argsort_higher_rank(self, shape):
        # Conv and patch-embedding weights have more than two dimensions; the
        # rank inversion must still run along dim 1.
        tensor = torch.randn(*shape)

        def reference(t):
            ranks = t.abs().argsort(dim=1).argsort(dim=1).to(torch.float32) + 1
            min_r = ranks.min(dim=1, keepdim=True).values
            max_r = ranks.max(dim=1, keepdim=True).values
            rank_norm = ((ranks - min_r) / (max_r - min_r)).clamp(0, 1)
            mask = torch.bernoulli((0.5 - 0.1) + rank_norm * 2 * 0.1)
            return t * mask

        torch.manual_seed(7)
        expected = reference(tensor)
        torch.manual_seed(7)
        result = sparsify(tensor, 0.5, SparsificationMethod.della_magprune, epsilon=0.1)
        assert result.shape == tensor.shape
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_half_precision_selection_matches_float_selection(self, dtype):
        # Half-precision tensors are selected through a histogram of their bit
        # patterns; check it agrees with the kthvalue path on the same values,
        # including zeros, signed zeros and subnormals.
        torch.manual_seed(6)
        base = (torch.randn(80, 50) * 0.02).to(dtype)
        tuned = (base.float() + torch.randn(80, 50) * 1e-3).to(dtype)
        delta = tuned - base
        delta[0, :5] = torch.tensor([0.0, -0.0, 1e-7, -1e-7, 6e-8]).to(dtype)
        w = delta.abs().reshape(-1)
        for k in (0, 1, 17, w.numel() // 2, w.numel() - 1, w.numel()):
            half_mask = _smallest_k_mask(w, k)
            float_mask = _smallest_k_mask(w.float(), k)
            assert int(half_mask.sum()) == k
            torch.testing.assert_close(half_mask, float_mask)
