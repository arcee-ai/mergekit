"""Batch execution is checked against independent per-output mathematics."""

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
    InputParameterTarget,
    MergeBatch,
    Option,
    ParameterScope,
    PerGroupValues,
    Shared,
    TensorBatch,
    TensorEntry,
    TensorGroup,
    TensorMetadata,
    from_batch_kernel,
    from_group_kernel,
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
        calls.append((batch.tensors.shape, batch.base_index))
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
        MergeBatch(tuple(groups)),
        weight=PerGroupValues(weights),
        normalize=PerGroupValues(normalize),
    )
    assert method.supports_batching
    assert [shape[0] for shape, _ in calls] == [2, 2, 1]
    for actual, wanted in zip(results.tensors, expected):
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
        MergeBatch.from_tensors(
            [torch.full((4,), float(i)), torch.full((4,), float(i + 2))]
        ).groups[0]
        for i in range(5)
    )
    result = method(MergeBatch(groups), weight=[0.5, 0.5], batch_options=options)
    assert [shape[0] for shape, _ in calls] == counts
    for i, tensor in enumerate(result.tensors):
        torch.testing.assert_close(tensor, torch.full((4,), float(i + 1)))


def _reference_slerp(a, b, t):
    # Deliberately scalar/CPU math, independent of the batch implementation.
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
    result = method(MergeBatch(tuple(groups)), t=PerGroupValues(ts))
    assert calls == [(torch.Size([6, 2, 2]), 0)]
    for actual, wanted in zip(result.tensors, expected):
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
    result = merge_methods.get(method_name)(MergeBatch(tuple(groups)), **params)
    for actual, wanted in zip(result.tensors, expected):
        torch.testing.assert_close(actual, wanted)


def test_non_base_coefficients_follow_canonical_layout(device):
    def kernel(
        batch: TensorBatch,
        weight: Annotated[
            torch.Tensor,
            BatchParameter(float, ParameterScope.INPUT, InputParameterTarget.NON_BASE),
        ],
    ) -> torch.Tensor:
        assert batch.base_index == 0
        return batch.tensors[:, 0] + (batch.tensors[:, 1:] * weight[:, :, None]).sum(1)

    method = from_batch_kernel(
        kernel, name="offsets", contract=InputContract(base=BasePolicy.REQUIRED)
    )
    base = TensorEntry("base", torch.tensor([10.0], device=device), True)
    a = TensorEntry("a", torch.tensor([2.0], device=device))
    b = TensorEntry("b", torch.tensor([3.0], device=device))
    groups = (TensorGroup((a, base, b)), TensorGroup((base, b, a)))
    result = method(MergeBatch(groups), weight={"a": 2, "b": 3})
    for tensor in result.tensors:
        torch.testing.assert_close(tensor, torch.tensor([23.0], device=device))


@pytest.mark.parametrize("failure", ["shape", "dtype", "coefficient"])
def test_later_invalid_group_fails_before_any_kernel(monkeypatch, failure):
    method = merge_methods.get("linear")
    calls = _record_calls(monkeypatch, method)
    good = MergeBatch.from_tensors([torch.ones(2), torch.ones(2)]).groups[0]
    bad_tensor = (
        torch.ones(3)
        if failure == "shape"
        else torch.ones(2, dtype=torch.float64 if failure == "dtype" else torch.float32)
    )
    bad = MergeBatch.from_tensors([torch.ones(2), bad_tensor]).groups[0]
    weight = (
        PerGroupValues([[0.5, 0.5], [0.5, "invalid"]])
        if failure == "coefficient"
        else [0.5, 0.5]
    )
    with pytest.raises(ValueError):
        method(MergeBatch((good, bad)), weight=weight)
    assert calls == []


def test_constraints_survive_both_signature_adapters():
    def group_kernel(
        group: TensorGroup, scale: Shared[Annotated[float, Field(gt=0)]]
    ) -> torch.Tensor:
        pytest.fail("Invalid value reached group kernel")

    def batch_kernel(
        batch: TensorBatch,
        scale: Annotated[torch.Tensor, BatchParameter(PositiveFloat)],
    ) -> torch.Tensor:
        pytest.fail("Invalid value reached batch kernel")

    batch = MergeBatch.from_tensors([torch.ones(1)])
    for factory, kernel in [
        (from_group_kernel, group_kernel),
        (from_batch_kernel, batch_kernel),
    ]:
        with pytest.raises(ValidationError):
            factory(kernel, name="positive")(batch, scale=-2)


def test_owned_workspace_does_not_mutate_sources():
    def kernel(batch: TensorBatch) -> torch.Tensor:
        return batch.workspace().add_(1).sum(1)

    source = torch.ones(3)
    method = from_batch_kernel(kernel, name="workspace")
    torch.testing.assert_close(
        method(MergeBatch.from_tensors([source])).one(), torch.full((3,), 2.0)
    )
    torch.testing.assert_close(source, torch.ones(3))
    borrowed = torch.ones(2, 1, 3)
    method.merge_batch(TensorBatch(borrowed))
    torch.testing.assert_close(borrowed, torch.ones_like(borrowed))


@pytest.mark.parametrize("method_name", ["linear", "slerp"])
def test_native_kernels_keep_autograd_and_borrowed_inputs(method_name, device):
    sources = [
        torch.tensor(v, device=device, requires_grad=True)
        for v in ([1.0, 2.0], [2.0, 4.0], [0.0, 3.0], [2.0, 1.0])
    ]
    copies = [v.detach().clone() for v in sources]
    groups = tuple(
        MergeBatch.from_tensors(sources[i : i + 2], base_index=0).groups[0]
        for i in (0, 2)
    )
    params = {"weight": [0.25, 0.75]} if method_name == "linear" else {"t": 0.75}
    result = merge_methods.get(method_name)(MergeBatch(groups), **params)
    sum(t.sum() for t in result.tensors).backward()
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
    result = merge_methods.get("slerp")(
        MergeBatch.from_tensors([a, b], base_index=0),
        t=0.5,
        batch_options=BatchOptions(max_bytes=1),
    ).one()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - initial
    # An oversized group still needs packed inputs and an output, but not
    # weight-sized float32 copies of every intermediate.
    assert peak <= a.nbytes + b.nbytes + result.nbytes + 32 * 1024 * 1024
    torch.testing.assert_close(result, torch.full_like(a, 1.5))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("normalize", [False, True])
def test_linear_precision_against_double_reference(dtype, normalize, device):
    a = torch.tensor([100.0, -10.0, 0.25, 3.0], device=device, dtype=dtype)
    b = torch.tensor([-50.0, 4.0, 3.0, -2.0], device=device, dtype=dtype)
    weights = [0.123456, 0.654321]
    expected = a.double() * weights[0] + b.double() * weights[1]
    if normalize:
        expected /= sum(weights)
    result = merge_methods.get("linear")(
        MergeBatch.from_tensors([a, b]), weight=weights, normalize=normalize
    ).one()
    torch.testing.assert_close(result, expected.to(dtype))


def test_group_fallback_preserves_metadata_without_packing(monkeypatch):
    def fail(*args):
        pytest.fail("Sequential fallback should not pack")

    monkeypatch.setattr("mergekit.merge_methods.batching.prepare_batches", fail)
    names = []

    def kernel(group: TensorGroup) -> torch.Tensor:
        names.append(group.metadata.name)
        return group.entries[0].tensor

    method = from_group_kernel(kernel, name="fallback")
    groups = tuple(
        TensorGroup((TensorEntry("a", torch.ones(i)),), TensorMetadata(name=str(i)))
        for i in (2, 3)
    )
    assert not method.supports_batching
    method(MergeBatch(groups))
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


def test_native_signature_rejects_ambiguous_annotations():
    def kernel(batch: TensorBatch, normalize: Shared[bool] = True) -> torch.Tensor:
        return batch.tensors.sum(1)

    with pytest.raises(TypeError, match="BatchParameter or Option"):
        from_batch_kernel(kernel, name="invalid")


def test_options_are_validated_before_execution():
    def kernel(batch: TensorBatch, modes: Option[list[str]]) -> torch.Tensor:
        pytest.fail("Unhashable execution options must fail before math")

    method = from_batch_kernel(kernel, name="options")
    with pytest.raises(TypeError, match="must be hashable"):
        method(MergeBatch.from_tensors([torch.ones(1)]), modes=["a"])


def test_empty_non_base_axis_has_a_dtype_without_validating_a_fake_value():
    def kernel(
        batch: TensorBatch,
        weight: Annotated[
            torch.Tensor,
            BatchParameter(
                PositiveFloat, ParameterScope.INPUT, InputParameterTarget.NON_BASE
            ),
        ],
    ) -> torch.Tensor:
        assert weight.shape == (2, 0)
        return batch.tensors[:, 0] + weight.sum(1, keepdim=True)

    method = from_batch_kernel(
        kernel, name="base_only", contract=InputContract(base=BasePolicy.REQUIRED)
    )
    group = MergeBatch.from_tensors([torch.ones(1)], base_index=0).groups[0]
    result = method(MergeBatch((group, group)))
    assert all(t.item() == 1 for t in result.tensors)


def test_non_base_coefficients_cannot_use_an_ignored_base_contract():
    def kernel(
        batch: TensorBatch,
        weight: Annotated[
            torch.Tensor,
            BatchParameter(float, ParameterScope.INPUT, InputParameterTarget.NON_BASE),
        ],
    ) -> torch.Tensor:
        return batch.tensors.sum(1)

    with pytest.raises(ValueError, match="base-aware"):
        from_batch_kernel(kernel, name="ambiguous_base")


def test_different_dtypes_and_input_counts_partition_batches(monkeypatch):
    method = merge_methods.get("linear")
    calls = _record_calls(monkeypatch, method)
    groups = tuple(
        MergeBatch.from_tensors([torch.ones(2, dtype=dtype)] * count).groups[0]
        for dtype, count in [
            (torch.float32, 2),
            (torch.float64, 2),
            (torch.float32, 3),
            (torch.float32, 2),
        ]
    )
    result = method(MergeBatch(groups), weight=1.0)
    assert [shape[:2] for shape, _ in calls] == [(2, 2), (1, 2), (1, 3)]
    assert [t.dtype for t in result.tensors] == [
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
            TensorMetadata(name="embedding", is_embed=True),
        )
        for rows in (4, 5)
    )
    with pytest.raises(ValueError, match="tokenizer.*source: base"):
        method(MergeBatch(groups), weight=[0.5, 0.5])
    assert calls == []


@pytest.mark.parametrize(
    "method_name,count",
    [("slerp", 2), ("nuslerp", 3), ("arcee_fusion", 2), ("model_stock", 3)],
)
def test_graph_adapter_preserves_optional_singleton_fallback(method_name, count):
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
    )
    tensor = torch.ones(2)
    assert task.execute({refs[0]: tensor}) is tensor
    if method_name == "model_stock":
        assert task.execute({refs[0]: tensor, refs[1]: tensor}) is None
