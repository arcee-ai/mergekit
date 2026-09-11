import pytest
import torch

from mergekit import merge_methods
from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods import BasePolicy, merge_state_dicts
from mergekit.merge_methods.arcee_fusion import DynamicThresholdFusion
from mergekit.merge_methods.task_adapter import ExecuteMergeMethodTask


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ]
)
def device(request):
    return request.param


@pytest.mark.parametrize("row_wise", [False, True])
def test_nuslerp_mixed_collinear_rows_have_finite_gradients(device, row_wise):
    a = torch.tensor([[1.0, 0.0]] * 4, dtype=torch.float64, device=device)
    b = torch.tensor(
        [[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
        device=device,
    )
    t = 0.25
    expected = torch.tensor(
        [[1.0, 0.0], [1.25, 0.0], [0.5, 0.0], [0.9238795325, 0.3826834324]],
        dtype=torch.float64,
        device=device,
    )
    if row_wise:
        a, b, expected = a.T, b.T, expected.T
    a.requires_grad_()
    b.requires_grad_()
    result = merge_state_dicts(
        [{"w": a}, {"w": b}],
        "nuslerp",
        parameters={
            "weight": [1 - t, t],
            "nuslerp_flatten": False,
            "nuslerp_row_wise": row_wise,
        },
    )["w"]
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    for tensor, coefficient in ((a, 1 - t), (b, t)):
        assert tensor.grad.isfinite().all()
        grad = tensor.grad.T if row_wise else tensor.grad
        # The first three rows take the linear fallback, independently of the
        # fourth row's spherical interpolation.
        torch.testing.assert_close(grad[:3], torch.full_like(grad[:3], coefficient))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_karcher_differentiates_output_scale(device, dtype):
    a = torch.tensor([1.0, 0.0], device=device, dtype=dtype, requires_grad=True)
    b = torch.tensor([2.0, 0.0], device=device, dtype=dtype, requires_grad=True)
    result = merge_state_dicts([{"w": a}, {"w": b}], "karcher")["w"]
    torch.testing.assert_close(result, a.new_tensor([1.5, 0.0]))
    result.sum().backward()
    for tensor in (a, b):
        assert tensor.grad.isfinite().all()
        torch.testing.assert_close(tensor.grad[0], tensor.new_tensor(0.5))


@pytest.mark.parametrize(
    "method_name,count,parameters",
    [
        ("nuslerp", 2, {"weight": [0.3, 0.7]}),
        ("nuslerp", 2, {"weight": [0.3, 0.7], "nuslerp_flatten": False}),
        ("karcher", 3, {"tol": 1e-8}),
    ],
)
def test_spherical_merge_gradients_match_finite_differences(
    device, method_name, count, parameters
):
    generator = torch.Generator().manual_seed(12)
    tensors = tuple(
        torch.randn(2, 3, generator=generator, dtype=torch.float64)
        .to(device)
        .requires_grad_()
        for _ in range(count)
    )

    def merge(*inputs):
        return merge_state_dicts(
            [{"w": tensor} for tensor in inputs], method_name, parameters=parameters
        )["w"]

    assert torch.autograd.gradcheck(merge, tensors)


@pytest.mark.parametrize("method_name", ["nuslerp", "multislerp"])
def test_optional_weight_cannot_silently_drop_a_configured_base(method_name):
    refs = tuple(ModelReference.model_validate(name) for name in ("base", "a", "b"))
    weight = WeightInfo(name="optional.bias", optional=True)
    task = ExecuteMergeMethodTask.from_parameters(
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

    baseless = task.model_copy(update={"base_index": None})
    torch.testing.assert_close(baseless.execute(tensors), torch.full((2,), 2**-0.5))


@pytest.mark.parametrize(
    "method_name",
    sorted(method.spec.name for method in merge_methods.registered_methods()),
)
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


def test_fusion_preserves_scalar_weight_shape():
    base = torch.tensor(1.0)
    other = torch.tensor(2.5)
    result = merge_state_dicts([{"w": base}, {"w": other}], "arcee_fusion", base=0)["w"]
    # A one-element softmax has zero KL divergence, so the zero threshold
    # selects the other input.
    torch.testing.assert_close(result, other)


def test_fusion_exact_quantiles_and_threshold():
    # Quantiles select the lower order statistic without interpolation.
    scores = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    fusion = DynamicThresholdFusion()
    torch.testing.assert_close(
        fusion.approximate_quantiles(scores, torch.tensor([0.25, 0.5, 0.75])),
        torch.tensor([2.0, 5.0, 8.0]),
    )
    assert fusion.calculate_dynamic_threshold(scores).item() == 14.0


def test_fusion_sampling_has_bounded_allocations(monkeypatch):
    # Sampling scratch scales with the sample size, not the population.
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


@pytest.mark.parametrize(
    "method_name",
    [
        "task_arithmetic",
        "ties",
        "dare_ties",
        "dare_linear",
        "breadcrumbs",
        "breadcrumbs_ties",
        "della",
        "della_linear",
    ],
)
@pytest.mark.parametrize("rescale", [False, True])
def test_task_arithmetic_values_and_gradients(method_name, rescale):
    from mergekit.merge_methods.generalized_task_arithmetic import get_mask
    from mergekit.sparsify import RescaleNorm, sparsify

    method = merge_methods.get(method_name)
    generator = torch.Generator().manual_seed(42)
    inputs = [
        torch.randn(4, 8, generator=generator, dtype=torch.float64, requires_grad=True)
        for _ in range(4)
    ]
    originals = [tensor.detach().clone() for tensor in inputs]
    weights = torch.tensor([0.7, -0.2, 0.5], dtype=torch.float64)[:, None, None]

    def reference():
        deltas = []
        for tensor in inputs[1:]:
            delta = tensor - inputs[0]
            if method.sparsification_method:
                delta = sparsify(
                    delta,
                    density=0.6,
                    method=method.sparsification_method,
                    rescale_norm=RescaleNorm.l1 if rescale else None,
                    gamma=0.01,
                    epsilon=0.15,
                )
            deltas.append(delta)
        weighted = torch.stack(deltas) * weights
        mask = (
            get_mask(weighted, method=method.consensus_method)
            if method.consensus_method
            else torch.ones_like(weighted)
        )
        divisor = (weights * mask).sum(0)
        divisor = torch.where(divisor.abs() < 1e-8, 1, divisor)
        return inputs[0] + 0.8 * (weighted * mask).sum(0) / divisor

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        expected = reference()
        expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
        torch.manual_seed(123)
        actual = merge_state_dicts(
            [{"w": tensor} for tensor in inputs],
            method,
            base=0,
            parameters={
                "weight": [0.7, -0.2, 0.5],
                "density": 0.6,
                "normalize": True,
                "rescale": rescale,
                "lambda": 0.8,
            },
        )["w"]
        actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    for tensor, original in zip(inputs, originals):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


@torch.inference_mode()
def test_task_arithmetic_peak_memory():
    # Inputs are borrowed; scratch should be one packed buffer plus a bounded
    # number of single-weight temporaries, not several copies of every delta.
    inputs = [{"w": torch.randn(256, 256)} for _ in range(17)]
    weight_bytes = inputs[0]["w"].nbytes
    with torch.profiler.profile(profile_memory=True) as profile:
        result = merge_state_dicts(
            inputs, "task_arithmetic", base=0, parameters={"weight": 1.0}
        )
    live = peak = 0
    for event in profile.profiler.kineto_results.events():
        if event.name() == "[memory]":
            live += event.nbytes()
            peak = max(peak, live)
    assert peak <= (16 + 3) * weight_bytes
    assert result["w"].isfinite().all()


@pytest.mark.parametrize(
    "method_name", ["linear", "slerp", "nuslerp", "task_arithmetic"]
)
def test_graph_and_state_dict_merges_agree(tmp_path, method_name):
    from safetensors.torch import save_file

    from mergekit.options import MergeOptions
    from mergekit.scripts.merge_raw_pytorch import (
        RawPyTorchMergeConfig,
        plan_flat_merge,
    )

    paths = []
    count = 3 if method_name == "nuslerp" else 2
    for i in range(count):
        path = tmp_path / f"{i}.safetensors"
        save_file({"w": torch.tensor([float(i), 1.0])}, path)
        paths.append(str(path))
    config = RawPyTorchMergeConfig(
        merge_method=method_name,
        models=[{"model": path} for path in paths],
        base_model=paths[-1],
        parameters={"weight": 0.5, "t": 0.25},
    )
    tasks = plan_flat_merge(config, str(tmp_path / "out"), False, False, MergeOptions())
    task = next(t.tensor_task for t in tasks if hasattr(t, "tensor_task"))
    tensors = {
        ref: torch.tensor([float(i), 1.0]) for i, ref in enumerate(task.model_order)
    }
    expected = merge_state_dicts(
        [{"w": tensor} for tensor in tensors.values()],
        method_name,
        base=count - 1,
        parameters={"t": 0.25} if method_name == "slerp" else {"weight": 0.5},
    )["w"]

    torch.testing.assert_close(task.execute(tensors), expected)


def test_optional_inputs_keep_their_original_coefficient_positions(monkeypatch):
    from mergekit.merge_methods import PerInput, TensorGroup, merge_method, registry

    @merge_method(name="optional_weighted_sum")
    def kernel(group: TensorGroup, weight: PerInput[float]) -> torch.Tensor:
        assert set(weight) == {entry.id for entry in group.entries}
        return sum(entry.tensor * weight[entry.id] for entry in group.entries)

    monkeypatch.setattr(registry, "_METHODS", {kernel.spec.name: kernel})
    refs = tuple(ModelReference.model_validate(name) for name in ("a", "missing", "c"))
    weight = WeightInfo(name="optional.bias", optional=True)
    task = ExecuteMergeMethodTask.from_parameters(
        method_name=kernel.spec.name,
        gather_tensors=GatherTensors(
            weight_info=ImmutableMap({ref: weight for ref in refs})
        ),
        model_order=refs,
        base_model=None,
        output_weight=weight,
        parameters=ImmutableMap({}),
        input_parameters=ImmutableMap(
            {
                ref: ImmutableMap({"weight": value})
                for ref, value in zip(refs, (2.0, 99.0, 3.0))
            }
        ),
    )
    actual = task.execute(
        {refs[0]: torch.tensor([10.0]), refs[2]: torch.tensor([20.0])}
    )
    torch.testing.assert_close(actual, torch.tensor([80.0]))


@pytest.mark.parametrize("failure", ["shape", "device"])
def test_graph_rejects_incompatible_loaded_tensors(failure):
    refs = tuple(ModelReference.model_validate(name) for name in ("a", "b"))
    info = WeightInfo(name="w")
    task = ExecuteMergeMethodTask.from_parameters(
        method_name="linear",
        gather_tensors=GatherTensors(
            weight_info=ImmutableMap({ref: info for ref in refs})
        ),
        model_order=refs,
        base_model=None,
        output_weight=info,
        parameters=ImmutableMap({"normalize": True}),
        input_parameters=ImmutableMap(
            {ref: ImmutableMap({"weight": 0.5}) for ref in refs}
        ),
    )
    other = torch.ones(3) if failure == "shape" else torch.ones(2, device="meta")
    with pytest.raises(ValueError, match="size mismatch|same device"):
        task.execute({refs[0]: torch.ones(2), refs[1]: other})
