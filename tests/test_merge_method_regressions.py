import pytest
import torch

from mergekit import merge_methods
from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods import BasePolicy, merge_state_dicts
from mergekit.merge_methods.arcee_fusion import DynamicThresholdFusion
from mergekit.merge_methods.task_adapter import ExecuteMergeMethodTask


@pytest.mark.parametrize("method_name", ["nuslerp", "multislerp"])
def test_optional_weight_cannot_silently_drop_a_configured_base(method_name):
    refs = tuple(ModelReference.model_validate(name) for name in ("base", "a", "b"))
    weight = WeightInfo(name="optional.bias", optional=True)
    task = ExecuteMergeMethodTask(
        method_name=method_name,
        gather_tensors=GatherTensors(
            weight_info=ImmutableMap({ref: weight for ref in refs})
        ),
        model_order=refs,
        base_model=refs[0],
        output_weight=weight,
        parameters=ImmutableMap({}),
        input_parameters=ImmutableMap(
            {ref: ImmutableMap({"weight": 0.5}) for ref in refs[1:]}
        ),
    )
    tensors = {
        refs[1]: torch.tensor([1.0, 0.0]),
        refs[2]: torch.tensor([0.0, 1.0]),
    }
    with pytest.raises(ValueError, match="Base input is not present.*optional.bias"):
        task.execute(tensors)

    # The same pair is valid when the user actually requested a baseless merge.
    baseless = task.model_copy(update={"base_model": None, "model_order": refs[1:]})
    torch.testing.assert_close(baseless.execute(tensors), torch.full((2,), 2**-0.5))


@pytest.mark.parametrize("method_name", sorted(merge_methods.REGISTERED_MERGE_METHODS))
def test_state_dict_methods_accept_strided_weights_without_mutation(method_name):
    _check_strided_weights(method_name)


@pytest.mark.parametrize("base", [False, True])
@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("row_wise", [False, True])
def test_strided_nuslerp_modes(base, flatten, row_wise):
    _check_strided_weights(
        "nuslerp",
        base=base,
        nuslerp_flatten=flatten,
        nuslerp_row_wise=row_wise,
    )


def test_strided_filter_wise_model_stock():
    _check_strided_weights("model_stock", filter_wise=True)


def _check_strided_weights(method_name, *, base=True, **overrides):
    method = merge_methods.get(method_name)
    base_id = (
        0
        if base
        and method.spec.contract.base in (BasePolicy.REQUIRED, BasePolicy.OPTIONAL)
        else None
    )
    count = 3 if base_id is not None else 2
    if method.spec.contract.max_inputs is not None:
        count = min(count, method.spec.contract.max_inputs)
    generator = torch.Generator().manual_seed(42)
    tensors = [torch.randn(4, 6, generator=generator).T for _ in range(count)]
    assert all(not tensor.is_contiguous() for tensor in tensors)
    copies = [tensor.clone() for tensor in tensors]
    values = {"weight": 1.0, "t": 0.4, "density": 0.7}
    parameters = {
        p.name: values[p.name] for p in method.spec.parameters if p.name in values
    }
    parameters.update(overrides)
    # DARE/DELLA are stochastic; compare layouts using the same random stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        expected = merge_state_dicts(
            [{"w": tensor.contiguous()} for tensor in tensors],
            method,
            base=base_id,
            parameters=parameters,
        )["w"]
        torch.manual_seed(123)
        actual = merge_state_dicts(
            [{"w": tensor} for tensor in tensors],
            method,
            base=base_id,
            parameters=parameters,
        )["w"]
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, expected)
    for tensor, before in zip(tensors, copies):
        torch.testing.assert_close(tensor, before)


def test_fusion_exact_quantiles_and_threshold():
    # Small inputs retain the original lower-order-statistic convention.
    scores = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    fusion = DynamicThresholdFusion()
    torch.testing.assert_close(
        fusion.approximate_quantiles(scores, torch.tensor([0.25, 0.5, 0.75])),
        torch.tensor([2.0, 5.0, 8.0]),
    )
    assert fusion.calculate_dynamic_threshold(scores).item() == 14.0


def test_fusion_sampling_has_bounded_allocations(monkeypatch):
    # Check allocations, not a particular random-number API: even a huge input
    # must use scratch proportional to the sample size, not the population.
    monkeypatch.setattr(
        "mergekit.merge_methods.arcee_fusion._QUANTILE_SAMPLE_SIZE", 1024
    )
    scores = torch.linspace(0, 1, 1_000_000)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        with torch.profiler.profile(profile_memory=True) as profile:
            quantiles = DynamicThresholdFusion().approximate_quantiles(
                scores, torch.tensor([0.25, 0.5, 0.75])
            )
    assert max(event.cpu_memory_usage for event in profile.events()) <= 64 * 1024
    torch.testing.assert_close(
        quantiles, torch.tensor([0.25, 0.5, 0.75]), atol=0.06, rtol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.inference_mode()
def test_fusion_releases_importance_scratch_before_thresholding(monkeypatch):
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn_like(a)
    initial = torch.cuda.memory_allocated()
    compute_mask = DynamicThresholdFusion.compute_fusion_mask

    def checked_mask(self, importance):
        # Only importance should remain, not full-sized diff/softmax buffers.
        assert torch.cuda.memory_allocated() - initial <= importance.nbytes + 1024**2
        return compute_mask(self, importance)

    monkeypatch.setattr(DynamicThresholdFusion, "compute_fusion_mask", checked_mask)
    result = merge_state_dicts([{"w": a}, {"w": b}], "arcee_fusion", base=0)["w"]
    assert result.isfinite().all()
