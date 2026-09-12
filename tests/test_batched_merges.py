import math
from typing import Annotated

import pytest
import torch
from pydantic import Field, PositiveFloat, ValidationError

from mergekit import merge_methods
from mergekit.merge_methods import (
    BasePolicy,
    BatchOptions,
    BatchParameter,
    InputContract,
    ParameterScope,
    PerGroupValues,
    TensorBatch,
    TensorEntry,
    TensorGroup,
    TensorMetadata,
    merge_method,
    merge_state_dicts,
)


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


def _record_calls(monkeypatch, method):
    calls = []
    implementation = method.implementation

    def record(batch, **kwargs):
        first = batch.tensors[0]
        shape = torch.Size((first.shape[0], len(batch.tensors), *first.shape[1:]))
        calls.append((shape, batch.base_index))
        return implementation(batch, **kwargs)

    monkeypatch.setattr(method, "implementation", record)
    return calls


def test_linear_buckets_options_shapes_and_restores_order(monkeypatch, device):
    method = merge_methods.get("linear")
    calls = _record_calls(monkeypatch, method)
    groups, expected = [], []
    weights, normalize = [], []
    for index, shape in enumerate([(3,), (2, 2), (3,), (3,), (2, 2)]):
        a = (
            torch.arange(math.prod(shape), device=device, dtype=torch.float32).reshape(
                shape
            )
            + index
        )
        b = a * 3 + 1
        # Input order and IDs vary between outputs, independently of shape buckets.
        entries = (TensorEntry("a", a), TensorEntry("b", b))
        if index % 2:
            entries = entries[::-1]
        groups.append(TensorGroup(entries))
        weight = {"a": index + 1.0, "b": 0.5}
        weights.append(weight)
        normalize.append(index != 3)
        result = a * weight["a"] + b * weight["b"]
        expected.append(result / sum(weight.values()) if normalize[-1] else result)
    results = method(
        tuple(groups),
        parameters={
            "weight": PerGroupValues(weights),
            "normalize": PerGroupValues(normalize),
        },
    )
    assert method.supports_batching
    assert [shape[0] for shape, _ in calls] == [2, 2, 1]
    for actual, wanted in zip(results, expected):
        torch.testing.assert_close(actual, wanted)
        assert actual.device.type == device


@pytest.mark.parametrize(
    "options,counts",
    [
        (BatchOptions(max_bytes=80), [2, 2, 1]),
        (BatchOptions(max_bytes=1), [1, 1, 1, 1, 1]),
        (BatchOptions(max_groups=3), [3, 2]),
    ],
)
def test_packing_budget_and_oversized_singletons(monkeypatch, options, counts):
    method = merge_methods.get("linear")
    calls = _record_calls(monkeypatch, method)
    # Each group packs 32 bytes of inputs and 8 bytes of coefficients.
    groups = tuple(
        TensorGroup.from_tensors(
            [torch.full((4,), float(i)), torch.full((4,), float(i + 2))]
        )
        for i in range(5)
    )
    result = method(groups, parameters={"weight": [0.5, 0.5]}, batch_options=options)
    assert [shape[0] for shape, _ in calls] == counts
    for i, tensor in enumerate(result):
        torch.testing.assert_close(tensor, torch.full((4,), float(i + 1)))


def _reference_slerp(a, b, t):
    x, y = a.double().cpu().reshape(-1), b.double().cpu().reshape(-1)
    nx, ny = x.norm().item(), y.norm().item()
    dot = sum(u * v for u, v in zip(x.tolist(), y.tolist())) / (
        (nx if nx > 1e-8 else 1) * (ny if ny > 1e-8 else 1)
    )
    if abs(dot) > 0.9995:
        result = (1 - t) * x + t * y
    else:
        theta = math.acos(max(-1, min(1, dot)))
        result = (math.sin((1 - t) * theta) * x + math.sin(t * theta) * y) / math.sin(
            theta
        )
    return result.reshape(a.shape).to(dtype=a.dtype, device=a.device)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("chunk_elements", [1024 * 1024, 5])
def test_slerp_mixed_geometries_varying_coefficients_and_base_order(
    monkeypatch, device, dtype, chunk_elements
):
    monkeypatch.setattr("mergekit.merge_methods.slerp._CHUNK_ELEMENTS", chunk_elements)
    method = merge_methods.get("slerp")
    calls = _record_calls(monkeypatch, method)
    pairs = [
        ([1.0, 0.0], [0.0, 2.0]),
        ([1.0, 2.0], [2.0, 4.0]),
        ([1.0, 2.0], [-1.0, -2.0]),
        ([0.0, 0.0], [1.0, 3.0]),
        ([2.0, -3.0], [4.0, 1.0]),
        ([0.0, 0.0], [0.0, 0.0]),
    ]
    ts = [-0.2, 0.25, 0.5, 0.8, 1.2, 1.0]
    groups, expected = [], []
    for index, ((a, b), t) in enumerate(zip(pairs, ts)):
        a, b = (torch.tensor(v, device=device, dtype=dtype) for v in (a, b))
        entries = (TensorEntry("base", a, is_base=True), TensorEntry("other", b))
        groups.append(TensorGroup(entries if index % 2 else entries[::-1]))
        expected.append(_reference_slerp(a, b, t))
    result = method(tuple(groups), parameters={"t": PerGroupValues(ts)})
    assert calls == [(torch.Size([6, 2, 2]), 0)]
    for actual, wanted in zip(result, expected):
        torch.testing.assert_close(actual, wanted)


@pytest.mark.parametrize("method_name", ["linear", "slerp"])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_native_kernels_preserve_weight_shape(method_name, shape, device):
    groups = []
    expected = []
    for i in range(3):
        a = torch.ones(shape, device=device) * (i + 1)
        b = a * 2
        if len(shape) == 2:
            a, b = a.T, b.T  # Packing also accepts noncontiguous sources.
        groups.append(
            TensorGroup((TensorEntry("base", a, True), TensorEntry("other", b)))
        )
        expected.append(a * 1.25)
    params = {"weight": [0.75, 0.25]} if method_name == "linear" else {"t": 0.25}
    result = merge_methods.get(method_name)(tuple(groups), parameters=params)
    for actual, wanted in zip(result, expected):
        torch.testing.assert_close(actual, wanted)


def test_non_base_coefficients_follow_canonical_layout(device):
    def kernel(
        batch: TensorBatch,
        weight: Annotated[
            torch.Tensor,
            BatchParameter(float, ParameterScope.NON_BASE),
        ],
    ) -> torch.Tensor:
        assert batch.base_index == 0
        return batch.tensors[0] + sum(
            tensor * coefficient[:, None]
            for tensor, coefficient in zip(batch.tensors[1:], weight.unbind(1))
        )

    method = merge_method(
        kernel, name="offsets", contract=InputContract(base=BasePolicy.REQUIRED)
    )
    base = TensorEntry("base", torch.tensor([10.0], device=device), True)
    a = TensorEntry("a", torch.tensor([2.0], device=device))
    b = TensorEntry("b", torch.tensor([3.0], device=device))
    groups = (TensorGroup((a, base, b)), TensorGroup((base, b, a)))
    result = method(groups, parameters={"weight": {"a": 2, "b": 3}})
    for tensor in result:
        torch.testing.assert_close(tensor, torch.tensor([23.0], device=device))


@pytest.mark.parametrize("failure", ["shape", "device", "coefficient"])
def test_later_invalid_group_fails_before_any_kernel(monkeypatch, failure):
    method = merge_methods.get("linear")
    calls = _record_calls(monkeypatch, method)
    good = TensorGroup.from_tensors([torch.ones(2), torch.ones(2)])
    bad_tensor = (
        torch.ones(3)
        if failure == "shape"
        else torch.ones(2, device="meta" if failure == "device" else "cpu")
    )
    bad = TensorGroup.from_tensors([torch.ones(2), bad_tensor])
    weight = (
        PerGroupValues([[0.5, 0.5], [0.5, "invalid"]])
        if failure == "coefficient"
        else [0.5, 0.5]
    )
    with pytest.raises(ValueError):
        method((good, bad), parameters={"weight": weight})
    assert calls == []


def test_constraints_survive_both_signature_adapters():
    def group_kernel(
        group: TensorGroup, scale: Annotated[float, Field(gt=0)]
    ) -> torch.Tensor:
        pytest.fail("Invalid value reached group kernel")

    def batch_kernel(
        batch: TensorBatch,
        scale: Annotated[torch.Tensor, BatchParameter(PositiveFloat)],
    ) -> torch.Tensor:
        pytest.fail("Invalid value reached batch kernel")

    batch = [TensorGroup.from_tensors([torch.ones(1)])]
    for kernel in (group_kernel, batch_kernel):
        with pytest.raises(ValidationError):
            merge_method(kernel, name="positive")(batch, parameters={"scale": -2})


@pytest.mark.parametrize("group_count", [1, 3])
def test_singleton_batches_borrow_strided_inputs(group_count):
    sources = [torch.arange(24.0).reshape(4, 6).T + i for i in range(2)]
    assert all(not source.is_contiguous() for source in sources)
    calls = []

    def kernel(batch: TensorBatch) -> torch.Tensor:
        for tensor, source in zip(batch.tensors, sources):
            assert tensor.shape == (1, *source.shape)
            assert tensor.data_ptr() == source.data_ptr()
            assert tensor.stride()[1:] == source.stride()
        calls.append(1)
        return sum(batch.tensors)

    method = merge_method(kernel, name="borrowed")
    group = TensorGroup.from_tensors(sources)
    results = method((group,) * group_count, batch_options=BatchOptions(max_groups=1))
    assert len(calls) == group_count
    for result in results:
        torch.testing.assert_close(result, sum(sources))


@pytest.mark.parametrize("method_name", ["linear", "slerp"])
def test_direct_kernels_accept_strided_inputs(method_name):
    sources = [torch.randn(2, 4, 6).transpose(1, 2) for _ in range(2)]
    parameters = (
        {"weight": torch.tensor([[0.25, 0.75]] * 2, dtype=torch.float64)}
        if method_name == "linear"
        else {"t": torch.tensor([0.25, 0.75], dtype=torch.float64)}
    )
    method = merge_methods.get(method_name)
    expected = method.merge_batch(
        TensorBatch(tuple(t.contiguous() for t in sources), base_index=0), **parameters
    )
    actual = method.merge_batch(TensorBatch(tuple(sources), base_index=0), **parameters)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("method_name", ["linear", "slerp"])
def test_native_kernels_keep_autograd_and_borrowed_inputs(method_name, device):
    sources = [
        torch.tensor(v, device=device, requires_grad=True)
        for v in ([1.0, 2.0], [2.0, 4.0], [0.0, 3.0], [2.0, 1.0])
    ]
    copies = [v.detach().clone() for v in sources]
    groups = tuple(
        TensorGroup.from_tensors(sources[i : i + 2], base_index=0) for i in (0, 2)
    )
    params = {"weight": [0.25, 0.75]} if method_name == "linear" else {"t": 0.75}
    result = merge_methods.get(method_name)(groups, parameters=params)
    sum(t.sum() for t in result).backward()
    for tensor, before in zip(sources, copies):
        torch.testing.assert_close(tensor, before)
        assert tensor.grad is not None and tensor.grad.isfinite().all()


def test_chunked_slerp_gradients(monkeypatch):
    from mergekit.merge_methods.slerp import slerp

    monkeypatch.setattr("mergekit.merge_methods.slerp._CHUNK_ELEMENTS", 4)
    generator = torch.Generator().manual_seed(123)
    a = torch.randn(2, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    b = torch.randn(2, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b, t: slerp(t, a, b), (a, b, t))

    zero = torch.zeros_like(a, requires_grad=True)
    slerp(t, zero, b).sum().backward()
    assert zero.grad.isfinite().all()
    assert b.grad.isfinite().all()
    assert t.grad.isfinite().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_slerp_large_weight_has_bounded_inference_scratch():
    a = torch.ones(4096, 2048, device="cuda", dtype=torch.bfloat16)
    b = torch.full_like(a, 2)
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    (result,) = merge_methods.get("slerp")(
        [TensorGroup.from_tensors([a, b], base_index=0)],
        parameters={"t": 0.5},
        batch_options=BatchOptions(max_bytes=1),
    )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - initial
    # Singleton inputs are borrowed; only output storage and bounded scratch remain.
    assert peak <= result.nbytes + 32 * 1024 * 1024
    torch.testing.assert_close(result, torch.full_like(a, 1.5))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("normalize", [False, True])
def test_linear_weighted_sum_and_normalization(dtype, normalize, device):
    a = torch.tensor([100.0, -10.0, 0.25, 3.0], device=device, dtype=dtype)
    b = torch.tensor([-50.0, 4.0, 3.0, -2.0], device=device, dtype=dtype)
    weights = [0.123456, 0.654321]
    expected = a.double() * weights[0] + b.double() * weights[1]
    if normalize:
        expected /= sum(weights)
    (result,) = merge_methods.get("linear")(
        [TensorGroup.from_tensors([a, b])],
        parameters={"weight": weights, "normalize": normalize},
    )
    torch.testing.assert_close(result, expected.to(dtype))


@pytest.mark.parametrize("method_name", ["linear", "slerp"])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_float_coefficients_follow_input_precision(
    monkeypatch, method_name, dtype, device
):
    method = merge_methods.get(method_name)
    implementation = method.implementation
    coefficient = "weight" if method_name == "linear" else "t"

    def checked(batch, **parameters):
        expected = torch.float64 if dtype == torch.float64 else torch.float32
        assert parameters[coefficient].dtype == expected
        assert parameters[coefficient].device == batch.tensors[0].device
        return implementation(batch, **parameters)

    monkeypatch.setattr(method, "implementation", checked)
    a = torch.tensor([1.0, 2.0], dtype=dtype, device=device)
    b = a * 2
    result = merge_state_dicts(
        [{"x": a, "y": a}, {"x": b, "y": b}],
        method,
        base=0,
        parameters={coefficient: [0.75, 0.25] if method_name == "linear" else 0.25},
    )
    for tensor in result.values():
        torch.testing.assert_close(tensor, a * 1.25)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("method_name", ["linear", "slerp"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_native_merges_on_mps(method_name, dtype):
    a = torch.tensor([1.0, 2.0], dtype=dtype, device="mps")
    result = merge_state_dicts(
        [{"x": a}, {"x": a * 2}],
        method_name,
        base=0,
        parameters={"weight": [0.75, 0.25]} if method_name == "linear" else {"t": 0.25},
    )
    torch.testing.assert_close(result["x"], a * 1.25)


@pytest.mark.parametrize("other_value", [1.0, 2.0])
def test_linear_zero_weight_sum_follows_tensor_division(device, other_value):
    source = torch.ones(3, device=device)
    batch = [TensorGroup.from_tensors([source, source * other_value])]
    method = merge_methods.get("linear")
    (result,) = method(batch, parameters={"weight": [1.0, -1.0]})
    expected = float("nan") if other_value == 1.0 else -float("inf")
    torch.testing.assert_close(
        result, torch.full_like(source, expected), equal_nan=True
    )
    (result,) = method(batch, parameters={"weight": [1.0, -1.0], "normalize": False})
    torch.testing.assert_close(result, torch.full_like(source, 1.0 - other_value))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("count", [2, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_linear_singleton_peak_memory(count, dtype):
    sources = [torch.ones(1024, 1024, device="cuda", dtype=dtype) for _ in range(count)]
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    (result,) = merge_methods.get("linear")(
        [TensorGroup.from_tensors(sources)], parameters={"weight": 1.0}
    )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - initial
    # One FP32 accumulator, also used as the output for FP32 inputs.
    budget = sources[0].numel() * 4
    if dtype != torch.float32:
        budget += result.nbytes
    assert peak <= budget + 4096
    torch.testing.assert_close(result, sources[0])


def test_linear_normalizes_before_narrowing_the_result(device):
    source = torch.full((4,), 60000.0, dtype=torch.float16, device=device)
    (result,) = merge_methods.get("linear")(
        [TensorGroup.from_tensors([source, source])], parameters={"weight": [1.0, 1.0]}
    )
    torch.testing.assert_close(result, source)


def test_group_method_receives_metadata_and_borrowed_inputs():
    names = []

    def kernel(group: TensorGroup) -> torch.Tensor:
        names.append(group.metadata.name)
        assert group.tensors[0] is groups[len(names) - 1].tensors[0]
        return group.entries[0].tensor

    method = merge_method(kernel, name="unpacked")
    groups = tuple(
        TensorGroup((TensorEntry("a", torch.ones(i)),), TensorMetadata(name=str(i)))
        for i in (2, 3)
    )
    assert not method.supports_batching
    method(groups)
    assert names == ["2", "3"]


def test_module_buffers_are_not_numerically_merged():
    models = [torch.nn.BatchNorm1d(2), torch.nn.BatchNorm1d(2)]
    result = merge_state_dicts(models, "linear", parameters={"weight": [0.5, 0.5]})
    assert result["num_batches_tracked"].dtype == torch.int64
    assert result["num_batches_tracked"].item() == 0
    models[1].num_batches_tracked += 1
    with pytest.raises(ValueError, match="Non-floating buffer"):
        merge_state_dicts(models, "linear", parameters={"weight": [0.5, 0.5]})


def test_state_dict_per_group_values_keep_alignment_across_buffers():
    models = [
        {
            "a": torch.tensor([1.0]),
            "counter": torch.tensor(2),
            "b": torch.tensor([3.0]),
        },
        {
            "a": torch.tensor([3.0]),
            "counter": torch.tensor(2),
            "b": torch.tensor([7.0]),
        },
    ]
    result = merge_state_dicts(
        models,
        "linear",
        parameters={"weight": PerGroupValues([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])},
    )
    assert list(result) == ["a", "counter", "b"]
    assert result["a"].item() == 1 and result["b"].item() == 7


def test_native_signature_rejects_group_annotations():
    from mergekit.merge_methods import PerInput

    def kernel(batch: TensorBatch, weight: PerInput[float]) -> torch.Tensor:
        return sum(batch.tensors)

    with pytest.raises(TypeError, match="Batch kernels use BatchParameter"):
        merge_method(kernel, name="invalid")


def test_options_are_validated_before_execution():
    def kernel(batch: TensorBatch, modes: list[str]) -> torch.Tensor:
        pytest.fail("Unhashable execution options must fail before math")

    method = merge_method(kernel, name="options")
    with pytest.raises(TypeError, match="must be hashable"):
        method([TensorGroup.from_tensors([torch.ones(1)])], parameters={"modes": ["a"]})


@pytest.mark.parametrize("scope", list(ParameterScope))
@pytest.mark.parametrize("value", [-(2**63) - 1, 2**63])
def test_integer_coefficients_overflow_before_any_kernel(scope, value):
    def kernel(
        batch: TensorBatch,
        coefficient: Annotated[torch.Tensor, BatchParameter(int, scope)],
    ) -> torch.Tensor:
        pytest.fail("An overflowing later coefficient must fail before any kernel")

    method = merge_method(
        kernel,
        name="integer_coefficients",
        contract=InputContract(base=BasePolicy.REQUIRED),
    )
    group = TensorGroup.from_tensors([torch.ones(1), torch.ones(1)], base_index=0)
    with pytest.raises(ValueError, match="coefficient.*torch.int64"):
        method(
            (group, group),
            parameters={"coefficient": PerGroupValues([0, value])},
            batch_options=BatchOptions(max_groups=1),
        )


@pytest.mark.parametrize("scope", list(ParameterScope))
def test_integer_coefficient_boundaries_pack_exactly(scope):
    def kernel(
        batch: TensorBatch,
        coefficient: Annotated[torch.Tensor, BatchParameter(int, scope)],
    ) -> torch.Tensor:
        assert coefficient.dtype == torch.int64
        assert (coefficient[0] == -(2**63)).all()
        assert (coefficient[1] == 2**63 - 1).all()
        return batch.tensors[0]

    method = merge_method(
        kernel,
        name="integer_coefficients",
        contract=InputContract(base=BasePolicy.REQUIRED),
    )
    group = TensorGroup.from_tensors([torch.ones(1), torch.ones(1)], base_index=0)
    results = method(
        (group, group),
        parameters={"coefficient": PerGroupValues([-(2**63), 2**63 - 1])},
    )
    assert len(results) == 2


def test_constrained_non_base_coefficients_accept_empty_axis():
    def kernel(
        batch: TensorBatch,
        weight: Annotated[
            torch.Tensor,
            BatchParameter(PositiveFloat, ParameterScope.NON_BASE),
        ],
    ) -> torch.Tensor:
        assert weight.shape == (2, 0)
        return batch.tensors[0] + weight.sum(1, keepdim=True)

    method = merge_method(
        kernel, name="base_only", contract=InputContract(base=BasePolicy.REQUIRED)
    )
    group = TensorGroup.from_tensors([torch.ones(1)], base_index=0)
    result = method((group, group))
    assert all(t.item() == 1 for t in result)


def test_non_base_coefficients_cannot_use_an_ignored_base_contract():
    def kernel(
        batch: TensorBatch,
        weight: Annotated[
            torch.Tensor,
            BatchParameter(float, ParameterScope.NON_BASE),
        ],
    ) -> torch.Tensor:
        return sum(batch.tensors)

    with pytest.raises(ValueError, match="base-aware"):
        merge_method(kernel, name="ambiguous_base")


def test_different_dtypes_and_input_counts_partition_batches(monkeypatch):
    method = merge_methods.get("linear")
    calls = _record_calls(monkeypatch, method)
    groups = tuple(
        TensorGroup.from_tensors([torch.ones(2, dtype=dtype)] * count)
        for dtype, count in [
            (torch.float32, 2),
            (torch.float64, 2),
            (torch.float32, 3),
            (torch.float32, 2),
        ]
    )
    result = method(groups, parameters={"weight": 1.0})
    assert [shape[:2] for shape, _ in calls] == [(2, 2), (1, 2), (1, 3)]
    assert [t.dtype for t in result] == [
        torch.float32,
        torch.float64,
        torch.float32,
        torch.float32,
    ]


def test_embedding_mismatch_fails_before_bucketing(monkeypatch):
    method = merge_methods.get("linear")
    calls = _record_calls(monkeypatch, method)
    groups = tuple(
        TensorGroup(
            (
                TensorEntry("a", torch.ones(rows, 2)),
                TensorEntry("b", torch.full((3, 2), 3.0)),
            ),
            TensorMetadata(name="embedding", vocabulary_axis=0),
        )
        for rows in (4, 5)
    )
    with pytest.raises(ValueError, match="tokenizer.*source: base"):
        method(groups, parameters={"weight": [0.5, 0.5]})
    assert calls == []


@pytest.mark.parametrize(
    "method_name,count",
    [
        ("slerp", 2),
        ("nuslerp", 3),
        ("arcee_fusion", 2),
        ("model_stock", 3),
        ("nearswap", 2),
    ],
)
@pytest.mark.parametrize(
    "dtype,out_dtype",
    [(None, None), (None, "float64"), ("bfloat16", None), ("bfloat16", "float64")],
)
def test_graph_adapter_preserves_optional_singleton_fallback(
    method_name, count, dtype, out_dtype
):
    from mergekit.architecture import WeightInfo
    from mergekit.common import ImmutableMap, ModelReference
    from mergekit.io.tasks import GatherTensors
    from mergekit.merge_methods.task_adapter import ExecuteMergeMethodTask

    refs = tuple(
        ModelReference.model_validate(name) for name in ["base", "a", "b"][:count]
    )
    weight = WeightInfo(name="optional.bias", optional=True)
    task = ExecuteMergeMethodTask(
        method_name=method_name,
        gather_tensors=GatherTensors(
            weight_info=ImmutableMap({r: weight for r in refs})
        ),
        model_order=refs,
        base_model=refs[0],
        output_weight=weight,
        parameters=ImmutableMap({}),
        input_parameters=ImmutableMap({}),
        dtype=dtype,
        out_dtype=out_dtype,
    )
    tensor = torch.tensor([1.003], requires_grad=True)
    expected = tensor.detach()
    for cast in (dtype, out_dtype):
        if cast is not None:
            expected = expected.to(getattr(torch, cast))
    result = task.execute({refs[0]: tensor})
    torch.testing.assert_close(result, expected)
    if dtype is None and out_dtype is None:
        assert result is tensor
    result.sum().backward()
    torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))
    buffer = torch.tensor(257)
    torch.testing.assert_close(task.execute({refs[0]: buffer}), buffer)
    if method_name == "model_stock":
        assert task.execute({refs[0]: tensor, refs[1]: tensor}) is None
        assert task.execute({refs[1]: tensor}) is None
    elif method_name == "nearswap":
        with pytest.raises(ValueError, match="Base input is not present"):
            task.execute({refs[1]: tensor})
        required = task.model_copy(
            update={"output_weight": weight.model_copy(update={"optional": False})}
        )
        with pytest.raises(ValueError, match="at least 2 inputs"):
            required.execute({refs[0]: tensor})
        with pytest.raises(ValueError, match="at least 2 inputs"):
            merge_methods.get(method_name).validate_inputs([refs[0]], refs[0])
    else:
        torch.testing.assert_close(task.execute({refs[1]: tensor}), expected)
